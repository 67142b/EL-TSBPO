"""Reuse isolated native planners instead of accumulating one library per task."""

from __future__ import annotations

_available_planners = []
_original_load_lib = None


def _install_planner_pool() -> None:
    import fast_downward

    global _original_load_lib
    if _original_load_lib is not None:
        return
    _original_load_lib = fast_downward.load_lib

    def acquire_planner():
        if _available_planners:
            return _available_planners.pop()
        return _original_load_lib()

    fast_downward.load_lib = acquire_planner


def make_textworld_env(gamefiles, request_infos, **kwargs):
    import textworld.gym

    _install_planner_pool()
    env_id = textworld.gym.register_games(gamefiles, request_infos, **kwargs)
    try:
        return textworld.gym.make(env_id)
    finally:
        # make() has already copied the construction arguments into the env.
        # Keeping every obsolete registration also retains wrappers/game lists.
        textworld.gym.registry.pop(env_id, None)


def close_textworld_env(environment) -> None:
    """Close a discarded batch and return its isolated planners to the pool.

    TextWorld's PddlEnv inherits a no-op close(). ctypes does not dlclose a
    discarded CDLL either, so Fast Downward's isolated memfd copies otherwise
    survive every task switch. Explicit dlclose also crashes this library's
    C++ exit handlers, so keep a bounded pool of clean, reusable instances.
    Capture the backends before batch.close() drops its children. This helper
    is for final disposal, not episode resets. Live backends never share a lib.
    """
    if environment is None:
        return
    pending = [environment]
    visited = set()
    backends = []
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        # Wrapper.__getattr__ forwards to the backend; inspect owned fields to
        # avoid seeing and unloading one library through several wrappers.
        fields = vars(current)
        if fields.get("downward_lib") is not None:
            backends.append(current)
        for name in ("batch_env", "_wrapped_env"):
            child = fields.get(name)
            if child is not None:
                pending.append(child)
        pending.extend(fields.get("envs", []))
    try:
        environment.close()
    finally:
        if backends:
            released = set()
            for backend in backends:
                library = backend.downward_lib
                backend.downward_lib = None
                if library is not None and library._handle not in released:
                    released.add(library._handle)
                    library.cleanup()
                    _available_planners.append(library)
