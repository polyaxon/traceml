from traceml import tracking


def callback(run=None):
    run = tracking.get_or_create_run(run)

    def _callback(env):
        res = {}
        for data_name, eval_name, value, _ in env.evaluation_result_list:
            key = data_name + "-" + eval_name
            res[key] = value
        run.log_metrics(step=env.iteration, **res)

    return _callback
