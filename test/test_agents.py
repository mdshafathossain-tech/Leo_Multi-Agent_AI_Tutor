"""Regression test: every tool exposed to an LLM must declare at least one argument.

Zero-argument tools produce a JSON schema with 'required' but no 'properties', which
providers like Groq reject with a 400 error."""
from src.agents import build_agents


def test_all_agent_tools_have_argument_schemas():
    for name, agent in build_agents().as_dict().items():
        for tool in agent.tools or []:
            schema_cls = getattr(tool, "args_schema", None)
            if schema_cls is None:
                continue
            props = schema_cls.model_json_schema().get("properties")
            assert props, f"{name}: tool '{getattr(tool, 'name', tool)}' has no arguments"