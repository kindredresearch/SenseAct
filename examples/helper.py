import builtins

def create_callback(shared_returns):
    builtins.shared_returns = shared_returns

    def kindred_callback(locals, globals):
        shared_returns = globals['__builtins__']['shared_returns']
        if locals['iters_so_far'] > 0:
            ep_rets = locals['seg']['ep_rets']
            ep_lens = locals['seg']['ep_lens']
            if len(ep_rets):
                if not shared_returns is None:
                    shared_returns['write_lock'] = True
                    shared_returns['episodic_returns'] += ep_rets
                    shared_returns['episodic_lengths'] += ep_lens
                    shared_returns['write_lock'] = False
    return kindred_callback
