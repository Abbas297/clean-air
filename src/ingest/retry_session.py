import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

def get_retry_session(
    retries: int = 4,
    backoff_factor: float = 1.5,
    status_forcelist: tuple = (429, 500, 502, 503, 504),
) -> requests.Session:
    """
    Returns a requests.Session configured with automatic retries and exponential backoff
    for transient network errors, rate limits (429), and server-side HTTP errors (500, 502, 503, 504).
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session
