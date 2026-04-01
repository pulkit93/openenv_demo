from openenv.core.env_server import create_app
from .environment import NumberGuessingEnvironment

# This is the exact pattern expected by the validator
app = create_app(NumberGuessingEnvironment)
