import random
import logging
import copy
import json
import json_repair
import time
from datetime import datetime

from data_utils import TemplateSample

from evaluate_utils import refuse_words
from env_utils import get_openai_client


logger = logging.getLogger("execute_step")


_OPENAI_CLIENT_CONFIG = {
    "openai_api_key": None,
    "openai_base_url": None,
}


def configure_openai_clients(
    *,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
) -> None:
    _OPENAI_CLIENT_CONFIG["openai_api_key"] = openai_api_key
    _OPENAI_CLIENT_CONFIG["openai_base_url"] = openai_base_url

# log_filename = f"logs/action_implementation/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# if not logger.handlers:
#     file_handler = logging.FileHandler(log_filename)
#     file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

#     console_handler = logging.StreamHandler()
#     console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

#     logger.addHandler(file_handler)
#     logger.addHandler(console_handler)

def call_llm(prompt):
    client = get_openai_client(
        openai_api_key=_OPENAI_CLIENT_CONFIG["openai_api_key"],
        openai_base_url=_OPENAI_CLIENT_CONFIG["openai_base_url"],
    )
    last_exception = None
    for _ in range(10):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-2024-11-20",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            if completion.choices and completion.choices[0].message.content.strip():
                return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"api call error: {e}")
            time.sleep(2)
            last_exception = e
    else:
        raise last_exception

def role_replace(template_sample: TemplateSample, component_pool: dict):
    new_role = random.choice(component_pool["role"])
    while new_role == template_sample.role:
        new_role = random.choice(component_pool["role"])
    template_sample.role = new_role

def env_replace(template_sample: TemplateSample, component_pool: dict):
    new_env = random.choice(component_pool["environment"])
    while new_env == template_sample.environment:
        new_env = random.choice(component_pool["environment"])
    template_sample.environment = new_env

def directive_replace(template_sample: TemplateSample, component_pool: dict):
    new_directive = random.choice(component_pool["directive"])
    while new_directive == template_sample.directive:
        new_directive = random.choice(component_pool["directive"])
    template_sample.directive = new_directive

def tips_replace_one(template_sample: TemplateSample, component_pool: dict):
    idx_to_replace = random.randint(0, len(template_sample.tips)-1)
    new_tip = random.choice(component_pool["tips"])
    while new_tip in template_sample.tips:
        new_tip = random.choice(component_pool["tips"])
    template_sample.tips[idx_to_replace] = new_tip

def tips_reorder(template_sample: TemplateSample):
    if len(template_sample.tips) < 2:
        return template_sample.tips
    random.shuffle(template_sample.tips)

def role_generalize(template_sample: TemplateSample):
    rephrase_prompt = """
    Rewrite the following role description to make it more general and broadly applicable.

    Requirements:
    - Keep the "You are ..." style.
    - Preserve the idea that the agent is responsible for task completion.
    - Remove domain-specific or overly specialized details.
    - Do not mention any specific platform, industry, or niche scenario unless absolutely necessary.
    - Keep it concise, natural, and instruction-like.
    - Return only the rewritten role sentence.

    Role:
    {role}
    """
    # logger.debug(f"role_generalize input: {template_sample.role}")
    generalized_role = call_llm(rephrase_prompt.format(role=template_sample.role))
    # logger.debug(f"generalized role: {generalized_role}")
    template_sample.role = generalized_role.strip()
    # return generalized_role.strip()

def role_operationalize(template_sample: TemplateSample):
    rephrase_prompt = """
    Rewrite the following role description to make it more operational and execution-oriented.

    Requirements:
    - Keep the "You are ..." format.
    - Emphasize active task execution rather than passive identity.
    - Make the role sound responsible for carrying out tasks in the environment.
    - The rewritten role should remain general and reusable.
    - Keep it concise, ideally one sentence.
    - Return only the rewritten role sentence.

    Role:
    {role}
    """
    # logger.debug(f"role_operationalize input: {template_sample.role}")
    operationalized_role = call_llm(rephrase_prompt.format(role=template_sample.role))
    # logger.debug(f"operationalized role: {operationalized_role}")
    template_sample.role = operationalized_role.strip()
    # return operationalized_role.strip()

def env_emphasize_tool_usage(template_sample: TemplateSample):
    rephrase_prompt = """
    Rewrite the following execution environment description to place more emphasis on tool usage.

    Requirements:
    - Preserve the original environment meaning.
    - Keep the second-person perspective (e.g., "You are operating..."); do not switch to third-person descriptions like "This environment...".
    - Explicitly convey that the environment provides tools for task execution.
    - Naturally reflect that using these tools is a normal and important part of completing tasks.
    - Do NOT turn this into a directive or a list of rules.
    - Keep it concise and natural.
    - Avoid over-emphasizing tool usage with redundant or overly strong statements.
    - Do NOT always prepend labels like "Environment:"; if used, it should be optional and not systematic.
    - Return only the rewritten environment text.

    Environment:
    {environment}
    """

    # logger.debug(f"env_emphasize_tool_usage input: {template_sample.environment}")
    emphasized_tool_usage_env = call_llm(rephrase_prompt.format(environment=template_sample.environment))
    # logger.debug(f"emphasized_tool_usage_env: {emphasized_tool_usage_env}")
    template_sample.environment = emphasized_tool_usage_env.strip()
    # return emphasized_tool_usage_env.strip()

def env_tighten_scope(template_sample: TemplateSample):
    rephrase_prompt = """
    Rewrite the following execution environment description to make its scope more bounded and explicit.

    Requirements:
    - Preserve the idea that the agent operates within a provided environment.
    - Keep the second-person perspective (e.g., "You are operating..."); do not switch to third-person descriptions like "The agent..." or "This environment...".
    - Emphasize that only the provided tools, resources, and accessible context within the environment are available.
    - Keep this as an environment description rather than a directive.
    - Do not add unnecessary detail.
    - Keep it concise and natural.
    - Do NOT always prepend labels like "Environment:"; if used, it should be optional and not systematic.
    - Return only the rewritten environment text.

    Environment:
    {environment}
    """

    # logger.debug(f"env_tighten_scope input: {template_sample.environment}")
    tightened_scope_env = call_llm(rephrase_prompt.format(environment=template_sample.environment))
    # logger.debug(f"tightened_scope_env: {tightened_scope_env}")
    template_sample.environment = tightened_scope_env.strip()
    # return tightened_scope_env.strip()

def directive_strengthen_constraints(template_sample: TemplateSample):
    rephrase_prompt = """
    Rewrite the following directive to make its constraints more explicit and stronger.

    Requirements:
    - Preserve the original task-execution intent.
    - Strengthen the language around staying within the environment, using only available tools/resources, and respecting tool input requirements.
    - Do not make it overly verbose.
    - Keep it as a directive, not a role or environment description.
    - Return only the rewritten directive text.

    Directive:
    {directive}
    """
    # logger.debug(f"directive_strengthen_constraints input: {template_sample.directive}")
    strengthened_directive = call_llm(rephrase_prompt.format(directive=template_sample.directive))
    # logger.debug(f"strengthened_directive: {strengthened_directive}")
    template_sample.directive = strengthened_directive.strip()
    # return strengthened_directive.strip()

def directive_shorten(template_sample: TemplateSample):
    rephrase_prompt = """
    Rewrite the following directive to make it shorter while preserving its core execution requirements.

    Requirements:
    - Keep the main constraints and execution intent.
    - Remove redundancy, repetition, and unnecessary wording.
    - Do not weaken essential constraints.
    - Keep it concise and natural.
    - Return only the shortened directive text.

    Directive:
    {directive}
    """
    # logger.debug(f"directive_shorten input: {template_sample.directive}")
    shortened_directive = call_llm(rephrase_prompt.format(directive=template_sample.directive))
    # logger.debug(f"shortened_directive: {shortened_directive}")
    template_sample.directive = shortened_directive.strip()
    # return shortened_directive.strip()

def directive_make_stepwise(template_sample: TemplateSample):

    rephrase_prompt = """
    Rewrite the following directive so that it more clearly encourages step-by-step execution.

    Requirements:
    - Preserve the original directive meaning.
    - Emphasize sequential, step-by-step progress.
    - Do NOT expand or decompose the directive into explicit sub-steps or ordered lists.
    - Maintain the original level of abstraction (keep it high-level and concise).
    - Avoid introducing specific step details (e.g., “first…, then…, finally…”).
    - Keep it sounding like a practical task directive, not an explanation or reasoning lecture.
    - Return only the rewritten directive text.

    Directive:
    {directive}
    """
    
    # logger.debug(f"directive_make_stepwise input: {template_sample.directive}")
    stepwise_directive = call_llm(rephrase_prompt.format(directive=template_sample.directive))
    # logger.debug(f"stepwise_directive: {stepwise_directive}")
    template_sample.directive = stepwise_directive.strip()
    # return stepwise_directive.strip()

def tips_strengthen(template_sample: TemplateSample):
    rephrase_prompt = """
    Rewrite the following execution tips to make them stronger and more instruction-like.

    Requirements:
    - Preserve the meaning of each tip.
    - Make the wording firmer, clearer, and more action-oriented.
    - Do not add new constraints that are not already implied.
    - Keep each tip concise.
    - Return a JSON list of rewritten tips only.

    Tips:
    {tips_json}
    """
    strengthen_num = random.randint(1, len(template_sample.tips))
    strengthen_indices = random.sample(range(len(template_sample.tips)), strengthen_num)
    tips_to_strengthen = [template_sample.tips[idx] for idx in strengthen_indices]
    # logger.debug(f"tips_strengthen input: {tips_to_strengthen}")
    strengthened_tips = json_repair.loads(call_llm(rephrase_prompt.format(tips_json=json.dumps(tips_to_strengthen, ensure_ascii=False))).strip())
    # logger.debug(f"strengthened_tips: {strengthened_tips}")
    for idx in range(strengthen_num):
        if strengthened_tips[idx] not in template_sample.tips:
            template_sample.tips[strengthen_indices[idx]] = strengthened_tips[idx]
    # return json_repair.loads(strengthened_tips.strip())

def tips_concretize(template_sample: TemplateSample):
    rephrase_prompt = """
    Rewrite the following execution tips to make them more concrete and operational.

    Requirements:
    - Preserve the original meaning of each tip.
    - Make each tip more explicit and executable by clarifying conditions, inputs, or expected behavior.
    - Do NOT introduce any new tools, functions, examples, or domain-specific details that are not present in the original tip.
    - Do NOT fabricate concrete instances (e.g., function names, parameters, scenarios).
    - The goal is to clarify and slightly specify the original tip, not to extend it with new content.
    - Avoid abstract strategy talk.
    - Keep each tip concise and natural.
    - Return a JSON list of rewritten tips only.

    Tips:
    {tips_json}
    """

    concretize_num = random.randint(1, len(template_sample.tips))
    concretize_indices = random.sample(range(len(template_sample.tips)), concretize_num)
    tips_to_concretize = [template_sample.tips[idx] for idx in concretize_indices]
    # logger.debug(f"tips_concretize input: {sentences}")
    concretized_tips = json_repair.loads(call_llm(rephrase_prompt.format(tips_json=json.dumps(tips_to_concretize, ensure_ascii=False))).strip())
    # logger.debug(f"concretized_tips: {concretized_tips}")
    for idx in range(concretize_num):
        if concretized_tips[idx] not in template_sample.tips:
            template_sample.tips[concretize_indices[idx]] = concretized_tips[idx]
    # return json_repair.loads(concretized_tips.strip())

def tips_prune(template_sample: TemplateSample):
    rephrase_prompt = """
    Prune the following execution tips.

    Requirements:
    - Remove redundant, overlapping, overly generic, or low-value tips.
    - Keep the tips that are most useful for execution.
    - Preserve coverage of distinct important guidance when possible.
    - If two tips are similar, keep the stronger or clearer one.
    - Do NOT compress, generalize, or remove important details within a single tip.
    - Only make minimal edits for clarity if necessary; preserve the original level of specificity.
    - The goal is to select and organize tips, not to rewrite or simplify their internal content.
    - Return a JSON list of the final retained tips only.

    Tips:
    {tips_json}
    """

    # logger.debug(f"tips_prune input: {template_sample.tips}")
    template_sample.tips = json_repair.loads(call_llm(rephrase_prompt.format(tips_json=json.dumps(template_sample.tips, ensure_ascii=False))).strip())
    # logger.debug(f"pruned_tips: {pruned_tips}")
    # return json_repair.loads(pruned_tips.strip())


if __name__ == "__main__":
    role_fn = [role_generalize, role_operationalize]
    env_fn = [env_emphasize_tool_usage, env_tighten_scope]
    directive_fn = [directive_strengthen_constraints, directive_shorten, directive_make_stepwise] 
    tips_fn = [tips_strengthen, tips_concretize, tips_prune]
    
    with open("data/initial_pool/agentharm/agentharm_component_initial_pool.json") as f:
        component_pool = json.load(f)
    
    role_pool = component_pool["role"]
    for role in role_pool:
        mutated_role = role
        for _ in range(3):
            fn = random.choice(role_fn)
            mutated_role = fn(mutated_role)

    environment_pool = component_pool["environment"]
    for env in environment_pool:
        mutated_env = env
        for _ in range(3):
            fn = random.choice(env_fn)
            mutated_env = fn(mutated_env)

    directive_pool = component_pool["directive"]
    for directive in directive_pool:
        mutated_directive = directive
        for _ in range(5):
            fn = random.choice(directive_fn)
            mutated_directive = fn(mutated_directive)

    tips_pool = component_pool["tips"]
    tips_subset = []
    for _ in range(20):
        subset_size = random.randint(3, 5)
        tips_subset.append(random.sample(tips_pool, subset_size))

    for subset in tips_subset:
        mutated_tips = subset
        for _ in range(5):
            fn = random.choice(tips_fn)
            if fn != tips_prune:
                logger.debug(f"Original tips: {mutated_tips}")
                mutate_num = random.randint(1, len(mutated_tips))
                mutate_indices = random.sample(range(len(mutated_tips)), mutate_num)
                tips_after_mutation = fn([mutated_tips[idx] for idx in mutate_indices])
                for idx in range(len(tips_after_mutation)):
                    if tips_after_mutation[idx] not in mutated_tips:
                        mutated_tips[mutate_indices[idx]] = tips_after_mutation[idx]
                logger.debug(f"Mutated tips: {mutated_tips}")
            else:
                logger.debug(f"Original tips: {mutated_tips}")
                mutated_tips = fn(mutated_tips)
                logger.debug(f"Pruned tips: {mutated_tips}")
