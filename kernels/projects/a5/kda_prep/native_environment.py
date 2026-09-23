"""Environment provenance embedded in every raw native receipt."""
from pathlib import Path
import hashlib
import os
import platform


def collect():
    import torch
    import torch_npu
    import ascriptor

    cann = Path(os.environ['ASCEND_HOME_PATH'])
    opp = Path(os.environ['ASCEND_OPP_PATH'])
    versions = {}
    for label, path in (('compiler', cann / 'compiler/version.info'),
                        ('opp', opp / 'version.info')):
        data = path.read_bytes()
        versions[label] = {'raw_lines': data.decode().splitlines(),
                           'sha256': hashlib.sha256(data).hexdigest()}
    return dict(python=platform.python_version(), torch=torch.__version__,
                torch_npu=torch_npu.__version__, ascriptor=ascriptor.__version__,
                device_sharing_authorized=os.environ.get('BF09_SHARED_DEVICE_AUTHORIZED') == '1',
                device_lock_scope=('task' if os.environ.get('BF09_SHARED_DEVICE_AUTHORIZED') == '1'
                                   else 'not_reported'),
                library_commit='90cfcdc720bbcd66e8bd4361c4dd4fbc1a2a57b5',
                kernels_commit='b3b3f9c16df7c4626ed3c081032a1be5a753d0b1',
                cann_version_records=versions,
                opp_directories=sorted(p.name for p in
                    (opp / 'built-in/op_impl/ai_core/tbe/kernel/config').iterdir() if p.is_dir()))
