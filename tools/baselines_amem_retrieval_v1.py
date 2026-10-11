"""One fresh retrieval over an immutable terminal A-Mem v4 bank.

No build, vector copy, corpus encoding, prompt change, or scientific-function
change. The only private binding redirects the original controller's writes.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
from pathlib import Path
from types import FunctionType
import numpy as np

from tools import baselines_amem_v4 as frozen
from tools.baselines_common_v1 import BaselineFatal, read, sha

FROZEN_SHA='0e9f85c2d9f2f48682af4c1e56dd68e9b5175a47a9ac1fcc5cc6cdf71189d6fb'
SCHEMA='amem-single-retrieval-existing-bank-v1'


def retrieve(question, state_path, output_dir, api, *, state_sha256, embeddings_sha256):
    if sha(frozen.__file__)!=FROZEN_SHA:raise BaselineFatal('Frozen v4 adapter changed')
    state_path=Path(state_path).resolve();output_dir=Path(output_dir).resolve()
    if sha(state_path)!=state_sha256:raise BaselineFatal('Terminal source state changed')
    original=read(state_path)
    source=Path(original['work_dir']).resolve();vectors=source/'cache/embeddings.npy'
    if state_path!=source/'state.json' or not original.get('build_complete'):
        raise BaselineFatal('Only the original complete terminal state can be restored')
    if output_dir==source or output_dir.is_relative_to(source) or source.is_relative_to(output_dir):
        raise BaselineFatal('Retrieval output must be separate from the immutable bank')
    if (not vectors.is_file() or sha(vectors)!=embeddings_sha256
            or original.get('embeddings_sha256')!=embeddings_sha256):
        raise BaselineFatal('Original vectors required unchanged; corpus re-encoding is forbidden')
    # Validate existing numeric bytes before model initialization/provider work.
    # The original restore will load these same bytes, without copying a cache.
    try:
        frozen._runtime._checked_vectors(np.load(vectors,allow_pickle=False),len(original['corpus']))
    except Exception as exc:
        raise BaselineFatal('Original numeric embedding cache is invalid') from exc
    if output_dir.exists():raise FileExistsError('A retrieval continuation output already exists')
    output_dir.mkdir(parents=True,exist_ok=False)
    state=deepcopy(original);state['work_dir']=str(output_dir)
    query_key=frozen._v1._sha({'question':question,'memory_sha256':original['memory_sha256']})
    def restore(derived,request_api):
        if derived!=state or request_api is not api:raise BaselineFatal('Private restore identity differs')
        if sha(state_path)!=state_sha256 or sha(vectors)!=embeddings_sha256:
            raise BaselineFatal('Original bank changed before restore')
        system,controller,classes=frozen._runtime._restore(original,api)
        controller.work_dir=output_dir
        return system,controller,classes
    native=frozen._runtime._retrieve
    call=FunctionType(native.__code__,{**native.__globals__,'_restore':restore},native.__name__,native.__defaults__,native.__closure__)
    call.__kwdefaults__=native.__kwdefaults__
    try:
        result=call(question,state,api,query_key)
    finally:
        if sha(state_path)!=state_sha256 or sha(vectors)!=embeddings_sha256:
            raise BaselineFatal('Read-only terminal bank changed during retrieval')
    return result


def describe():
    return {'schema':SCHEMA,'frozen_adapter_sha256':FROZEN_SHA,
            'science':'original _restore plus unchanged _retrieve bytecode; only controller output directory redirected',
            'new_bank':False,'corpus_reencoding':False,'vector_copy':False,'writer_generation':False}
