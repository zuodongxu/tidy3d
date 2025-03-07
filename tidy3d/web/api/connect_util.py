"""connect util for webapi."""

import time
from functools import wraps

# from requests import ReadTimeout
# from requests.exceptions import ConnectionError as ConnErr
# from requests.exceptions import JSONDecodeError
# from urllib3.exceptions import NewConnectionError

from ...exceptions import WebError
from ...log import log

# number of seconds to keep re-trying connection before erroring
CONNECTION_RETRY_TIME = 180
# time between checking task status
REFRESH_TIME = 2


def wait_for_connection(decorated_fn=None, wait_time_sec: float = CONNECTION_RETRY_TIME):
    """Causes function to ignore connection errors and retry for ``wait_time_sec`` secs."""
    return decorated_fn


def get_time_steps_str(time_steps) -> str:
    """get_time_steps_str"""

    raise NotImplementedError("This function is not implemented yet.")

    if time_steps < 1000:
        time_steps_str = f"{time_steps}"
    elif 1000 <= time_steps < 1000 * 1000:
        time_steps_str = f"{time_steps / 1000}K"
    else:
        time_steps_str = f"{time_steps / 1000 / 1000}M"
    return time_steps_str


def get_grid_points_str(grid_points) -> str:
    """get_grid_points_str"""

    raise NotImplementedError("This function is not implemented yet.")

    if grid_points < 1000:
        grid_points_str = f"{grid_points}"
    elif 1000 <= grid_points < 1000 * 1000:
        grid_points_str = f"{grid_points / 1000}K"
    else:
        grid_points_str = f"{grid_points / 1000 / 1000}M"
    return grid_points_str
