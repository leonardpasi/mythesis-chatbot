import os

from trulens.core import TruSession
from trulens.dashboard import run_dashboard

tru = TruSession(database_url=os.getenv("SUPABASE_CONNECTION_STRING"))
run_dashboard(tru)
