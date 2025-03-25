import threading
import pandas as pd
from openbb import obb

# Thread-local storage for OpenBB client.
thread_local = threading.local()

def get_openbb_client():
    """
    Returns a thread-local OpenBB client instance.
    """
    if not hasattr(thread_local, "openbb_client"):
        thread_local.openbb_client = obb
    return thread_local.openbb_client

def get_attribute_value(obj, attr_name, default=0):
    """
    Safely extract an attribute (or dict key) value from an object.
    Returns default if not found or if value is NaN.
    """
    if hasattr(obj, attr_name):
        value = getattr(obj, attr_name)
        if value is not None and not pd.isna(value):
            return value
    try:
        if attr_name in obj:
            value = obj[attr_name]
            if value is not None and not pd.isna(value):
                return value
    except (TypeError, KeyError):
        pass
    try:
        value = obj[attr_name]
        if value is not None and not pd.isna(value):
            return value
    except (TypeError, KeyError, IndexError):
        pass
    return default

def openbb_has_technical():
    """
    Check if the technical module is available in the OpenBB client.
    
    Returns:
        bool: True if available, False otherwise
    """
    client = get_openbb_client()
    return hasattr(client, 'technical')