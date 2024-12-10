
def rerun_if_error(func):
    def inner(*args, **kwargs):
        for _ in range(3):
            try:
                func(*args, **kwargs)
                break
            except Exception as err:
                print("Error:", err)

    return inner
