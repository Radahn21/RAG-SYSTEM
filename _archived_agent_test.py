import os
from dotenv import load_dotenv
from azure.core.exceptions import HttpResponseError
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition

try:
    from .auth import get_credential
except ImportError:
    from auth import get_credential

 
load_dotenv()
 
project_client = AIProjectClient(
    endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
    credential=get_credential(),
)

try:
    agent = project_client.agents.create_version(
        agent_name="GeneralPurposeAssistant",
        definition=PromptAgentDefinition(
            model=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
            instructions="You are a helpful assistant that answers general questions",
        ),
    )


except HttpResponseError as exc:
    if "Tenant provided in token does not match resource token" in str(exc):
        raise SystemExit(
            "Azure authentication tenant mismatch. Sign in with an account in the AI Project tenant. "
            "If .env sets AZURE_TENANT_ID, update it to the project's tenant or leave "
            "AZURE_USE_EXPLICIT_TENANT unset so interactive sign-in can use the selected account tenant."
        ) from exc
    raise

from azure.identity import InteractiveBrowserCredential
cred = InteractiveBrowserCredential(tenant_id=os.environ["AZURE_TENANT_ID"])

print(f"Agent created (id: {agent.id}, name: {agent.name}, version: {agent.version})")