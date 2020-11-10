_ALGORITHMS = dict()


def register(name):

  def add_to_dict(fn):
    global _ALGORITHMS
    _ALGORITHMS[name] = fn
    return fn

  return add_to_dict


def get_algorithm(hparams, G, F, X, Y):
  if hparams.algorithm not in _ALGORITHMS:
    print(f'Algorithm {hparams.algorithm} not found')
    exit()

  return _ALGORITHMS[hparams.algorithm](hparams, G, F, X, Y)
