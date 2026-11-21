from types import SimpleNamespace

import inference._5_planned_local_weighted_budget_cap_model as base


class PlannedStepwiseBudgetDiag3_Model(
    base.PlannedLocalWeightedBudgetCap_Model
):
    """
    Additional diagnostic implementation:
    generate each sub-question separately, passing previous outputs forward.
    The original model is not modified.
    """

    @base.llm_retry(max_retries=10, default_output=("", 0))
    async def reasoning(self, query):
        schedule_map = {
            "constant": base.constant_schedule,
            "linear": base.linear_decay,
            "polynomial": base.polynomial_decay,
            "exponential": base.exponential_decay,
            "cosine": base.cosine_anneal,
        }

        schedule = schedule_map.get(self.decay.lower())
        if schedule is None:
            raise ValueError(f"Unsupported decay schedule: {self.decay}")

        total_budget = base.level_map[query.level][1]
        budgets = base.allocate_tokens(
            query.credits,
            total_budget,
            schedule,
        )

        system_prompt = self._prompt["system"].format(
            instruction=self.instruction
        )

        previous_results = []
        usage_totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        step_diagnostics = []
        self.last_step_diagnostics = []
        self.last_step_outputs = []

        for index, (step, allocated_budget) in enumerate(
            zip(query.steps, budgets), start=1
        ):
            # Keep each call's cap explicit. A zero allocation is raised
            # to one token so the step can still be attempted.
            step_cap = max(1, int(allocated_budget))

            prior_context = (
                "\n\n".join(previous_results)
                if previous_results
                else "No previous step results."
            )

            reference_parts = []
            if query.reference:
                reference_parts.append(str(query.reference))
            reference_parts.append(
                "Results from previously completed steps:\n"
                + prior_context
            )
            combined_reference = "\n\n".join(reference_parts)

            decomposed = (
                f"Current sub-question ({index}/{len(query.steps)}):\n"
                f"{step}\n\n"
                f"Use no more than approximately {step_cap} generated tokens "
                "for this step. Solve this step only, and make its result "
                "useful to subsequent steps."
            )

            user_message = self._prompt["user"].format(
                question=query.description,
                reference=combined_reference,
                level=query.level,
                decomposed=decomposed,
                output_format=self.output_format,
            )

            response = await base.generate_response(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=step_cap,
                return_raw=True,
                logger=self.logger,
            )

            step_output = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)

            prompt_tokens = int(
                getattr(usage, "prompt_tokens", 0) or 0
            )
            completion_tokens = int(
                getattr(usage, "completion_tokens", 0) or 0
            )
            total_tokens = int(
                getattr(usage, "total_tokens", prompt_tokens + completion_tokens)
                or prompt_tokens + completion_tokens
            )

            usage_totals["prompt_tokens"] += prompt_tokens
            usage_totals["completion_tokens"] += completion_tokens
            usage_totals["total_tokens"] += total_tokens

            step_diagnostics.append({
                "step": index,
                "sub_question": str(step),
                "output_text": step_output,
                "allocated_budget": int(allocated_budget),
                "actual_max_new_tokens": step_cap,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            })

            previous_results.append(
                f"Step {index}: {step}\n"
                f"Step result:\n{step_output}"
            )

            self.logger.debug(
                f"[Stepwise budget] step={index}, "
                f"allocated={allocated_budget}, cap={step_cap}, "
                f"prompt_tokens={prompt_tokens}, "
                f"completion_tokens={completion_tokens}\n"
                f"Output:\n{step_output}"
            )

        # Return all intermediate work so the existing evaluator can inspect
        # the final answer in the context of the previous steps.
        final_output = "\n\n".join(previous_results)

        # Preserve the usage fields expected by the existing process_question.
        aggregate_usage = SimpleNamespace(
            prompt_tokens=usage_totals["prompt_tokens"],
            completion_tokens=usage_totals["completion_tokens"],
            total_tokens=usage_totals["total_tokens"],
            step_diagnostics=step_diagnostics,
        )

        self.logger.debug(
            f"[Stepwise budget summary] {step_diagnostics}"
        )

        self.last_step_diagnostics = [dict(item) for item in step_diagnostics]
        self.last_step_outputs = [dict(item) for item in step_diagnostics]
        return final_output, aggregate_usage
