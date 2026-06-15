from dprt.evaluation.exporters.lh_pairs import build_lh_pairs
from dprt.evaluation.exporters.kradar import build_kradar


def build(name: str, *args, **kwargs):
    if name == 'lh_pairs':
        return build_lh_pairs(*args, **kwargs)
    if name == 'kradar':
        return build_kradar(*args, **kwargs)
    raise ValueError(f'Unsupported exporter: {name}')
