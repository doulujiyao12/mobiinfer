#!/usr/bin/python
import sys
import json
import os
import subprocess
import json
post_treat = {}
qnn_sdk = os.environ["QNN_SDK_ROOT"]
print(qnn_sdk)
with open(sys.argv[1]) as f:
    post_treat = json.load(f)
soc_id = int(sys.argv[2])
dsp_arch = sys.argv[3]
print('soc_id:', soc_id, "; dsp_arch:", dsp_arch)
qnn_bin_path = os.path.join(qnn_sdk, 'bin', 'x86_64-linux-clang')

# qnn-model-lib-generator computes its share dir from __file__'s parent.parent.parent.
# If a writable mirror exists at $QNN_MIRROR (or default ~/qnn-mirror) with a patched
# Makefile that puts weight blobs into .ldata via objcopy --rename-section, route
# through the mirror so the patched Makefile is used. This is REQUIRED for LLM-scale
# graphs whose weights would otherwise overflow .data/.bss past PC32 reach.
_mirror = os.environ.get('QNN_MIRROR') or os.path.expanduser('~/qnn-mirror')
_mirror_lib_gen = os.path.join(_mirror, 'bin', 'x86_64-linux-clang', 'qnn-model-lib-generator')
if os.path.isfile(_mirror_lib_gen):
    qnnModelLibGenerator = _mirror_lib_gen
    print(f"[npu_convert] Using QNN mirror with .ldata-patched Makefile: {_mirror}")
else:
    qnnModelLibGenerator = os.path.join(qnn_bin_path, 'qnn-model-lib-generator')
qnnContextBinaryGenerator = os.path.join(qnn_bin_path, 'qnn-context-binary-generator')

# LLM-scale graphs bake gigabytes of weights into .bss/.data; default x86 small
# code model can't reach data symbols beyond ±2GB (R_X86_64_PC32 truncation in
# system-provided crtstuff.o, which we cannot recompile).
#
# Fix: medium code model with -mlarge-data-threshold=65536 -- objects >=64KB
# (i.e. the weight blobs from objcopy'd .raw) go into .ldata with 64-bit
# addressing, while small symbols stay in .data with 32-bit addressing so
# crtstuff's PC32 references still fit. The QNN Makefile uses `CXXFLAGS +=` /
# `LDFLAGS +=` so these prepend cleanly into both compile and link commands.
# threshold=0 forces every static symbol into .ldata regardless of size, so
# .data shrinks to just crtstuff's tiny bookkeeping. Some QNN graphs (graph0
# in our case) generate enormous .text from many small addNode() calls, and
# the linker layout pushes .data/.bss past PC32 reach even though raw data
# itself is small (< 100MB). Forcing everything to .ldata keeps .data near .text.
_extra = '-mcmodel=medium -mlarge-data-threshold=0'
os.environ['CXXFLAGS'] = (_extra + ' ' + os.environ.get('CXXFLAGS', '')).strip()
os.environ['LDFLAGS']  = (_extra + ' ' + os.environ.get('LDFLAGS',  '')).strip()

# qnn-model-lib-generator hardcodes `make CXX=clang++`. If the system clang is
# too old to recognize -mlarge-data-threshold (needs clang >=17), prepend a
# user-provided shim dir that aliases clang++ -> g++. We auto-detect at common
# locations; users can override with QNN_CLANG_SHIM_DIR.
_shim_dir = os.environ.get('QNN_CLANG_SHIM_DIR')
if not _shim_dir:
    for candidate in [os.path.expanduser('~/bin/qnn-shim'),
                      '/root/bin/qnn-shim']:
        if os.path.isfile(os.path.join(candidate, 'clang++')):
            _shim_dir = candidate
            break
if _shim_dir and _shim_dir not in os.environ.get('PATH', '').split(os.pathsep):
    os.environ['PATH'] = _shim_dir + os.pathsep + os.environ.get('PATH', '')
    print(f"[npu_convert] Prepended clang++ shim dir to PATH: {_shim_dir}")
merges = post_treat["merge"]
cache_dir = 'res'
if 'cache' in post_treat:
    cache_dir = post_treat['cache']
clean_tmp = True
context_config = {
    "backend_extensions": {
        "shared_library_path": os.path.join(qnn_sdk, "lib","x86_64-linux-clang","libQnnHtpNetRunExtensions.so"),
        "config_file_path": "./htp_backend_extensions.json"
    }
}
htp_so = os.path.join(qnn_sdk, 'lib','x86_64-linux-clang','libQnnHtp.so')
with open('context_config.json', 'w') as f:
    f.write(json.dumps(context_config, indent=4))

htp_backend_extensions = {
    "graphs": [
        {
            "vtcm_mb": 8,
            "O": 3.0,
            "fp16_relaxed_precision": 1,
            "hvx_threads": 4
        }
    ],
    "devices": [
        {
            "soc_id": soc_id,
            "dsp_arch": dsp_arch,
            "cores": [
                {
                    "core_id": 0,
                    "perf_profile": "burst",
                    "rpc_control_latency": 100
                }
            ]
        }
    ],
    "context": {
        "weight_sharing_enabled": True
    }
}

for key in post_treat["merge"]:
    srcs = merges[key]
    dst = key
    dstname = key.split('/')
    dstname = dstname[len(dstname)-1]
    dstname = dstname.replace('.bin', '')
    graphs = []
    libs = []
    workdirs = []
    for i,src in enumerate(srcs):
        # tar
        graphname = src.split('/')
        graphname = graphname[len(graphname)-1]
        graphs.append(graphname)
        workdir = os.path.join(os.getcwd(), src)
        workdirs.append(workdir)
        print(subprocess.run("tar -cf " + graphname + '.bin' + ' *.raw', cwd=workdir, capture_output=True, text=True, shell=True))
        if clean_tmp:
            print(subprocess.run('rm *.raw', cwd=workdir, capture_output=True, text=True, shell=True))
        # Compile
        compile_cmd = 'python3 ' + qnnModelLibGenerator + ' -c ' + os.path.join(workdir, graphname + '.cpp') + ' -b ' + os.path.join(workdir, graphname + '.bin') + ' -t x86_64-linux-clang -o ' + workdir
        # Use subprocess with check so a link failure (e.g. PC32 truncation)
        # aborts immediately instead of silently continuing through every other
        # subgraph (each takes minutes to compile).
        compile_rc = subprocess.run(compile_cmd, shell=True).returncode
        out_so = os.path.join(workdir, 'x86_64-linux-clang', 'lib' + graphname + '.so')
        if compile_rc != 0 or not os.path.exists(out_so):
            sys.exit(f"[npu_convert] qnn-model-lib-generator failed for {graphname} "
                     f"(rc={compile_rc}, .so missing={not os.path.exists(out_so)})")
        if clean_tmp:
            os.popen("rm " + os.path.join(workdir, graphname + '.bin')).read()
        libs.append(out_so)
    htp_backend_extensions['graphs'][0]['graph_names'] = graphs
    with open('htp_backend_extensions.json', 'w') as f:
        f.write(json.dumps(htp_backend_extensions, indent=4))
    libsStr = ""
    for i in range(0, len(libs)):
        if i > 0:
            libsStr+=','
        libsStr += libs[i]
    print(os.popen(qnnContextBinaryGenerator + ' --model ' + libsStr + ' --backend '+ htp_so + ' --binary_file ' + dstname + ' --config_file ./context_config.json ' + ' --output_dir ' + cache_dir).read())
    if clean_tmp:
        for workdir in workdirs:
            os.popen("rm -rf " + workdir).read()



