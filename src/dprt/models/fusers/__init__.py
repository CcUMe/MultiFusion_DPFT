from dprt.models.fusers.basempfusion import build_basempfusion
from dprt.models.fusers.mpfusion import build_basempoptfusion

def build_fuser(name: str, *args, **kwargs):
    if 'mpfusion' in name.lower():
        return build_basempoptfusion(*args, **kwargs)

    if 'basefusion' in name.lower():
        return build_basempfusion(*args, **kwargs)

    if 'baseoptfusion' in name.lower():
        return build_basempoptfusion(*args, **kwargs)

    raise ValueError(f'Unsupported fuser: {name}')
