import os

from trulens.core import TruSession
from trulens.dashboard import run_dashboard

tru = TruSession(database_url=os.getenv("SUPABASE_PROD_CONNECTION_STRING_IPV4"))
run_dashboard(tru)
