import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import Cache
import copy

from .decoding import DecodingConfig


class SpecDec:
    def __init__(self, model_path, device_map, is_encoder_decoder=False, is_instruct=False, gamma=10, offload_path=None):
        self.model_path = model_path
        _AutoModel = AutoModelForSeq2SeqLM if is_encoder_decoder else AutoModelForCausalLM
        self.target_model = _AutoModel.from_pretrained(model_path,
                                                       torch_dtype=torch.float16,
                                                       device_map=device_map,
                                                       trust_remote_code=False,
                                                       offload_folder=offload_path
                                                       )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokens_per_step = gamma
        self.is_encoder_decoder = is_encoder_decoder
        self.is_instruct = is_instruct


    @staticmethod
    def _force_token(logits, token_id):
        # Overwrite `logits` in-place to put all mass on `token_id`.
        # Shape: (..., V). Pass a slice (e.g., logits[..., 0, :]) to clamp a single timestep.
        logits.fill_(-float("inf"))
        logits[..., token_id] = 0


    def _build_model_inputs(self, input_tokens, attn_mask, kv_cache,
                            fixed_gating, logits_to_keep,
                            encoder_outputs=None, encoder_attn_mask=None,
                            unfinished_sequences=None):
        '''
            Build the kwargs dict for target_model.forward(). Handles both causal and
            encoder-decoder models. For encoder-decoder, `encoder_outputs` is deep-copied
            and its `last_hidden_state` sliced to `unfinished_sequences` so the enclosing
            loop can drop finished sequences without mutating the original.
        '''
        inputs = {
            "past_key_values": kv_cache,
            "use_cache": True,
            "fixed_gating": fixed_gating,
            "logits_to_keep": logits_to_keep,
        }
        if self.is_encoder_decoder:
            enc_out = copy.deepcopy(encoder_outputs)
            enc_out.last_hidden_state = encoder_outputs.last_hidden_state[unfinished_sequences]
            inputs["input_ids"] = None
            inputs["decoder_input_ids"] = input_tokens
            inputs["encoder_outputs"] = enc_out
            inputs["attention_mask"] = encoder_attn_mask[unfinished_sequences]
            inputs["decoder_attention_mask"] = attn_mask
        else:
            inputs["input_ids"] = input_tokens
            inputs["attention_mask"] = attn_mask
            inputs["position_ids"] = attn_mask.cumsum(dim=-1)[:, -input_tokens.size(-1):] - 1
        return inputs


    def evaluate(self, output_tokens, px, qx, num_valid_tokens, greedy=False):
        '''
            Apply the speculative-decoding acceptance test to each draft token.
            Input tensors
                - output_tokens    : (b, n)       — draft tokens to verify
                - px               : (b, n+1, V)  — target log-probs
                - qx               : (b, n, V)    — draft log-probs
                - num_valid_tokens : (b, 1)       — number of pre-EOS draft tokens per row
            Output
                - num_accepted     : (b, 1)       — number of leading accepted draft tokens per row
        '''
        batch_size = qx.size(0)
        if greedy:
            ground_truth = torch.argmax(torch.exp(px), dim=-1)
            is_accepted = ground_truth[..., :-1] == output_tokens
        else:
            token_idx = output_tokens.unsqueeze(-1)                                  # (b, n, 1)
            p_i = px[:, :output_tokens.size(-1), :].gather(-1, token_idx).squeeze(-1)  # (b, n)
            q_i = qx.gather(-1, token_idx).squeeze(-1)                                # (b, n)
            ratio = torch.exp(p_i - q_i)
            r_i = torch.rand_like(ratio)
            is_accepted = r_i <= ratio

        no_rejection_yet = (~is_accepted).cumsum(dim=-1) == 0  # True while all preceding tokens accepted
        num_accepted = no_rejection_yet.sum(dim=-1, keepdim=True)
        num_accepted = torch.min(num_accepted.view(-1, 1), num_valid_tokens)

        return num_accepted


    def resample(self, reject_pos, px, qx, greedy=False):
        '''
            Input tensors
                - reject_pos: (b, 1)
                - px: (b, n+1, V)
                - qx: (b, n, V)
            Output
                - resampled_token: (b, 1)
        '''

        gather_index_reject_pos = reject_pos.expand(-1, px.size(-1)).unsqueeze(1)
        if greedy:
            prob = torch.exp(px.gather(1, gather_index_reject_pos).squeeze(1))
            resampled_token = torch.argmax(prob, dim=-1).unsqueeze(-1)
        else:
            # resample the last token
            all_accepted = (reject_pos == qx.size(1)).squeeze(1)
            target_log_probs = px.gather(1, gather_index_reject_pos).squeeze(1).float()
            draft_log_probs = qx.new_full(
                (qx.size(0), qx.size(-1)), float('-inf'), dtype=torch.float32
            )
            draft_log_probs[~all_accepted] = qx[~all_accepted].gather(
                1, gather_index_reject_pos[~all_accepted]
            ).squeeze(1).float()

            # Residual sampling uses max(p - q, 0). In low precision the
            # residual can have zero total mass (for example when p and q
            # round to the same distribution). Compute in fp32 and fall back
            # to p in that case to avoid an all-NaN softmax.
            residual = (target_log_probs.exp() - draft_log_probs.exp()).clamp_min_(0)
            residual_mass = residual.sum(dim=-1, keepdim=True)
            valid_residual = torch.isfinite(residual_mass) & (residual_mass > 0)
            safe_mass = torch.where(valid_residual, residual_mass, torch.ones_like(residual_mass))
            adjust_dist = residual / safe_mass
            target_dist = torch.nn.functional.softmax(target_log_probs, dim=-1)
            adjust_dist = torch.where(valid_residual, adjust_dist, target_dist)
            resampled_token = torch.multinomial(adjust_dist, num_samples=1)

        return resampled_token


    def speculate(self, model_inputs, forced_bos_token_id=None, temperature=1.0,
                  decoding_config=None):
        '''
            Run up to `self.tokens_per_step` (gamma) draft steps using the target model with
            fixed gating. Mutates `model_inputs` in place to advance past_key_values, input_ids,
            attention_mask, and position_ids each step. Stops early if every sequence has
            produced EOS.
            Output
                - output_probs   : (b, n, V)  — log-probs for each draft step (n <= gamma)
                - output_tokens  : (b, n)     — sampled draft tokens
                - draft_kv_cache                  — KV cache after the final draft step
        '''
        decoding = decoding_config or DecodingConfig(
            mode="greedy" if temperature == 0 else "sampling",
            temperature=temperature if temperature > 0 else 1.0,
        )
        input_ids_attr = "decoder_input_ids" if self.is_encoder_decoder else "input_ids"
        attn_mask_attr = "decoder_attention_mask" if self.is_encoder_decoder else "attention_mask"
        num_sequences = model_inputs[input_ids_attr].size(0)

        eos_token_id = self.tokenizer.eos_token_id
        draft_kv_cache = model_inputs["past_key_values"]

        output_probs = tuple()
        output_tokens = tuple()
        finished_sequences = torch.zeros(num_sequences).bool()

        for _ in range(self.tokens_per_step):
            if torch.all(finished_sequences):
                break

            draft_outputs = self.target_model.forward(**model_inputs)

            logits = draft_outputs.logits[..., -1, :]
            if draft_kv_cache is None and forced_bos_token_id is not None:
                self._force_token(logits, forced_bos_token_id)

            log_probs = decoding.log_probs(logits)
            if decoding.greedy:
                draft_inputs = torch.argmax(logits, dim=-1).unsqueeze(-1)
            else:
                draft_inputs = torch.multinomial(log_probs.exp(), num_samples=1)

            output_probs = output_probs + (log_probs,)
            output_tokens = output_tokens + (draft_inputs,)
            draft_kv_cache = draft_outputs.past_key_values

            eos_idx = (draft_inputs == eos_token_id).squeeze(1)
            finished_sequences[eos_idx] = True

            # update model inputs
            model_inputs[input_ids_attr] = draft_inputs
            model_inputs["past_key_values"] = draft_kv_cache
            if model_inputs[attn_mask_attr] is not None:
                attention_mask = model_inputs[attn_mask_attr]
                model_inputs[attn_mask_attr] = torch.cat((attention_mask, attention_mask.new_ones(attention_mask.size(0), 1)), dim=-1)
                if not self.is_encoder_decoder:
                    model_inputs["position_ids"] = model_inputs[attn_mask_attr].cumsum(dim=-1)[:, -1:] - 1

        output_probs = torch.stack(output_probs, dim=1)
        output_tokens = torch.cat(output_tokens, dim=-1)

        return output_probs, output_tokens, draft_kv_cache


    def update_attention_mask(self, draft_attn_mask, num_accepted, new_length):
        '''
            Input tensors
                - draft_attn_mask: (num_valid_seq, past_kv_length)
                - num_accepted   : (num_valid_seq, 1)
                - new_length     : Number
            Output tensor
                - draft_attn_mask: (num_valid_seq, past_kv_length + new_length + 1)
        '''
        new_draft_attn_mask = torch.arange(0, new_length, device=draft_attn_mask.device).expand(draft_attn_mask.size(0), new_length) < num_accepted
        draft_attn_mask = torch.cat((draft_attn_mask, new_draft_attn_mask, draft_attn_mask.new_ones(draft_attn_mask.size(0), 1)), dim=-1)
        return draft_attn_mask


    def update_kv_caches(self, kv_cache, past_kv_length, valid_indices):
        '''
            Trim the shared KV cache to `past_kv_length` positions and keep only the rows in
            `valid_indices` (dropping sequences that just finished). Handles both tuple-style
            caches (legacy) and transformers.Cache objects.
        '''
        if isinstance(kv_cache, tuple):
            new_kv_cache = tuple([(
                                        layer_kv[0][valid_indices, :, :past_kv_length, :],
                                        layer_kv[1][valid_indices, :, :past_kv_length, :],
                                    ) + (
                                        (
                                            layer_kv[2][valid_indices, ...],
                                            layer_kv[3][valid_indices, ...],
                                        ) if self.is_encoder_decoder else tuple()
                                    )
                                    for layer_kv in kv_cache])
            return new_kv_cache
        elif isinstance(kv_cache, Cache):
            kv_cache.batch_select_indices(valid_indices)
            kv_cache.crop(past_kv_length)
            return kv_cache
        raise TypeError(f"unexpected kv_cache type: {type(kv_cache)}")


    def speculative_decoding(self,
                             prompt,
                             system_prompt=None,
                             forced_bos_token_id=None,
                             input_length=128,
                             max_tokens=256,
                             temperature=1.0,
                             decoding_config=None):
        '''
            End-to-end speculative decoding loop. Tokenizes `prompt`, then repeatedly:
              (1) drafts gamma tokens with fixed gating,
              (2) verifies them with full gating in a single target forward,
              (3) accepts the longest prefix that passes the ratio test and resamples the
                  next token from the residual distribution.
            Continues until every sequence emits EOS or reaches `max_tokens`.
            Output
                - dict with keys: input_ids, output_ids, output_text, num_steps, num_generated
        '''
        assert self.target_model is not None

        decoding = decoding_config or DecodingConfig(
            mode="greedy" if temperature == 0 else "sampling",
            temperature=temperature if temperature > 0 else 1.0,
        )

        max_gen = max_tokens - input_length

        tokenizer = self.tokenizer
        eos_token_id = tokenizer.eos_token_id

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if not self.is_encoder_decoder and not self.is_instruct:
            system_prompt = "" if system_prompt is None else system_prompt
            prompt = ["[INST] " + system_prompt + p for p in prompt]

        if self.is_instruct:
            def chat_prompt(p):
                return [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": p,
                    }
                ]
            tokenizer_output = tokenizer.apply_chat_template([chat_prompt(p) for p in prompt],
                                                             return_dict=True,
                                                             tokenize=True,
                                                             return_tensors="pt",
                                                             add_generation_prompt=True,
                                                             max_length=input_length,
                                                             padding=True,
                                                             truncation=True,
                                                             )
        else:
            tokenizer_output = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=input_length)

        input_ids = tokenizer_output.input_ids                       # (b, input_len) on cpu
        if not self.is_encoder_decoder and not self.is_instruct:
            INST_END = tokenizer(r"[\INST]", return_tensors="pt", add_special_tokens=False).input_ids
            input_ids[:, -INST_END.size(-1):]  = INST_END
        input_ids = input_ids.cuda()                                   # (b, input_len) on cuda
        attention_mask = tokenizer_output.attention_mask.cuda()        # (b, input_len) on cuda

        # encoder
        encoder_outputs = None
        encoder_attn_mask = None
        if self.is_encoder_decoder:
            encoder = self.target_model.get_encoder()
            encoder_attn_mask = attention_mask                         # (b, input_len) on cuda
            encoder_outputs = encoder(input_ids, return_dict=True, attention_mask=encoder_attn_mask)
            input_ids = torch.ones((input_ids.size(0), 1), dtype=torch.long).cuda()  # (b, 1) on cuda
            input_ids[:, 0] = self.target_model.config.decoder_start_token_id

            target_attn_mask = torch.ones_like(input_ids)              # (b, 1) on cuda
            draft_attn_mask = torch.ones_like(input_ids)               # (b, 1) on cuda

        else:
            target_attn_mask = attention_mask                          # (b, input_len) on cuda
            draft_attn_mask = attention_mask                           # (b, input_len) on cuda

        kv_cache = None
        draft_inputs = input_ids  # (b, input_len) on cuda for causal; (b, 1) on cuda for enc-dec

        num_steps = 0
        unfinished_sequences = torch.arange(input_ids.size(0))         # (b,) on cpu
        avg_num_tokens_generated = []
        total_generated = torch.zeros((input_ids.size(0), 1))          # (b, 1) on cpu
        total_generated_tokens = [[] for _ in range(input_ids.size(0))]

        is_first_step = True
        while len(unfinished_sequences):

            ### 1. Generate draft sequences

            if isinstance(kv_cache, Cache):
                _seq_length = kv_cache.get_seq_length()

            model_inputs = self._build_model_inputs(
                draft_inputs, draft_attn_mask, kv_cache,
                fixed_gating=True, logits_to_keep=1,
                encoder_outputs=encoder_outputs, encoder_attn_mask=encoder_attn_mask,
                unfinished_sequences=unfinished_sequences,
            )

            draft_probs, draft_tokens, _ = self.speculate(model_inputs,
                                                          forced_bos_token_id=forced_bos_token_id,
                                                          decoding_config=decoding)
            # draft_probs:  (b, n, V) on cuda — draft log-probs, n <= gamma
            # draft_tokens: (b, n) on cuda    — sampled draft tokens

            '''
                Compute `num_valid_tokens`, which contains the number of valid draft tokens for each sequence
                e.g., for eos_token_id == 2,
                    [ 9, 79,  2,  2,  2] --> 3
                    [23, 59,  6, 91, 34] --> 5
                    [48, 29, 73, 92,  2] --> 5
                    [ 2,  2,  2,  2,  2] --> 1
            '''
            num_spec_tokens = draft_tokens.size(-1)
            is_eos = (draft_tokens == eos_token_id)
            has_eos = is_eos.any(dim=1)
            first_eos_idx = is_eos.int().argmax(dim=1)            # first EOS position (0 when absent)
            first_eos_idx[~has_eos] = num_spec_tokens - 1         # no EOS → all tokens valid
            num_valid_tokens = (first_eos_idx + 1).view(-1, 1)    # (b, 1) on cuda


            ### 2. Verify draft sequences with target model

            '''
                valid_token_mask example
                    [ 9, 79,  2,  2,  2]     [  True,  True,  True, False, False]
                    [23, 59,  6, 91, 34] --> [  True,  True,  True,  True,  True]
                    [48, 29, 73, 92,  2]     [  True,  True,  True,  True,  True]
            '''
            valid_token_mask = (torch.arange(num_spec_tokens).expand(draft_tokens.size(0), num_spec_tokens) \
                                    .to(num_valid_tokens.device) < num_valid_tokens) \
                                    .to(draft_tokens.dtype)
            valid_draft_tokens = draft_tokens * valid_token_mask

            '''
                `draft_inputs` carries exactly the tokens verify needs as prefix in every iteration:
                on the first step it holds the full prompt (since the kv_cache is empty and verify must
                re-see the whole prefix), and on later steps it holds just the previously resampled token
                (a single token, since all earlier accepted tokens are already in the shared kv_cache).
            '''
            target_inputs = torch.cat((draft_inputs, valid_draft_tokens), dim=-1)
            target_attn_mask = torch.cat((draft_attn_mask, valid_token_mask), dim=-1)

            if isinstance(kv_cache, Cache):
                kv_cache.crop(_seq_length)

            model_inputs = self._build_model_inputs(
                target_inputs, target_attn_mask, kv_cache,
                fixed_gating=False, logits_to_keep=num_spec_tokens + 1,
                encoder_outputs=encoder_outputs, encoder_attn_mask=encoder_attn_mask,
                unfinished_sequences=unfinished_sequences,
            )

            verify_outputs = self.target_model.forward(**model_inputs)

            logits = verify_outputs.logits
            if is_first_step and forced_bos_token_id is not None:
                self._force_token(logits[..., 0, :], forced_bos_token_id)

            target_probs = decoding.log_probs(logits)[:, -num_spec_tokens-1:]  # (b, n+1, V) on cuda
            kv_cache = verify_outputs.past_key_values


            ### 3. Accept and resample with probabilities
            greedy = decoding.greedy
            num_accepted = self.evaluate(draft_tokens, target_probs, draft_probs, num_valid_tokens, greedy=greedy)  # (b, 1) on cuda
            accepted_tokens = draft_tokens * (torch.arange(num_spec_tokens, device=draft_tokens.device).expand(draft_tokens.size(0), num_spec_tokens) < num_accepted)  # (b, n) on cuda
            resampled_tokens = self.resample(num_accepted, target_probs, draft_probs, greedy=greedy)  # (b, 1) on cuda
            contains_eos = (accepted_tokens == eos_token_id).any(dim=1)  # (b,) on cuda

            max_accepted = num_accepted.max()                         # scalar (0-d tensor)
            final_tokens = accepted_tokens.new_zeros(num_accepted.size(0), max_accepted + 1)  # (b, max_accepted+1) on cuda
            final_tokens[:, :max_accepted] = accepted_tokens[:, :max_accepted]
            need_resample = ~contains_eos                            # (b,) — rows that didn't hit EOS
            final_tokens[need_resample, num_accepted[need_resample, 0]] = resampled_tokens[need_resample, 0]

            ### 4. Prepare next step

            num_generated = num_accepted.clone()                       # (b, 1) on cuda
            num_generated[~contains_eos] += 1

            # A speculative step may accept several tokens at once. Limit the
            # final step per sequence to the caller's exact generation budget
            # instead of returning up to `gamma` extra tokens.
            remaining = max_gen - total_generated[unfinished_sequences]
            remaining = remaining.to(
                device=num_generated.device, dtype=num_generated.dtype
            ).clamp_min(0)
            num_generated = torch.minimum(num_generated, remaining)

            kept_token_mask = (
                torch.arange(final_tokens.size(1), device=final_tokens.device).unsqueeze(0)
                < num_generated
            )
            kept_contains_eos = ((final_tokens == eos_token_id) & kept_token_mask).any(dim=1)

            total_generated[unfinished_sequences] += num_generated.to(total_generated.device)
            avg_num_tokens_generated += num_generated.flatten().tolist()
            for i, seq_id in enumerate(unfinished_sequences.tolist()):
                total_generated_tokens[seq_id] += final_tokens[i, :num_generated[i, 0]].tolist()

            finished = kept_contains_eos.to(unfinished_sequences.device)
            finished = finished | (total_generated[unfinished_sequences, 0] >= max_gen)
            unfinished_sequences = unfinished_sequences[~finished]


            '''
                The resampled token is the only position not yet in the shared kv_cache: after verify, the cache
                covers prefix + all accepted tokens (the last accepted token's K,V was computed by verify with
                full gating). From the draft model's own perspective the last accepted token would be "unseen"
                (the draft's own forward never processed it into a K,V) — but since draft and target share the
                cache in SpecMoE, verify's K,V is reused and only the resampled token needs to be fed to the
                next draft phase.
            '''
            draft_inputs = final_tokens.gather(1, (num_generated-1).view(-1, 1))[~finished]  # (b', 1) on cuda

            # Update attention mask and kv cache
            draft_attn_mask = self.update_attention_mask(draft_attn_mask[~finished], num_accepted[~finished, :], max_accepted)
            past_kv_length = draft_attn_mask.size(-1) - 1
            kv_cache = self.update_kv_caches(kv_cache, past_kv_length, ~finished)

            num_steps += 1
            is_first_step = False

        output_text = tokenizer.batch_decode(total_generated_tokens)

        return {
                    "input_ids": tokenizer_output.input_ids.tolist(),
                    "output_ids": total_generated_tokens,
                    "output_text": output_text,
                    "num_steps": num_steps,
                    "num_generated": sum(avg_num_tokens_generated) / len(avg_num_tokens_generated)
                }
