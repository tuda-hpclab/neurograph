# Bundled dependencies

`cpp-utility` and `mpi-wrapper` are header-only libraries developed alongside this
project at the Technical University of Darmstadt. They are not published as separate
packages, so their headers ship with this repository instead of being downloaded at
configure time.

| Directory | Upstream | Revision |
| --- | --- | --- |
| [`cpp-utility/`](cpp-utility/) | `https://git.rwth-aachen.de/parallel/cpp-utilities` | `2bfcb47080bb7f3d5206574276ec0e6415b5a988` |
| [`mpi-wrapper/`](mpi-wrapper/) | `https://git.rwth-aachen.de/parallel/mpi-wrapper` | `71948e8a7b5151670d84babddff57c356f534a78` |

Both carry the same BSD 3-Clause license and the same copyright holder as this
repository, see [`../LICENSE.dat`](../LICENSE.dat). Their contributors are part of
[`../AUTHORS.dat`](../AUTHORS.dat).

The contents of each directory are the verbatim `include/<name>` tree of the revision
above; nothing was edited. To update one, replace the directory with the new upstream
`include/<name>` and record the new revision here.

`external/` is added as a single `SYSTEM` include directory in
[`../cmake/ProjectLibraries.cmake`](../cmake/ProjectLibraries.cmake), so every
`#include <cpp-utility/...>` and `#include <mpi-wrapper/...>` resolves unchanged, and
the warning set of this project — which differs from the one these libraries are built
with upstream — does not apply to them.
