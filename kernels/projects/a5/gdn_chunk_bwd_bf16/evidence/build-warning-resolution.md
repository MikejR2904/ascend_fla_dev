# BF-02 vendor compilation warnings

Scope: six actual CCE vendor builds at block_dim2 (unchanged FP32 checkpoint,
reverse, group_reduce and new BF16 equivalents), library90cfcdc, CANN9.2.0.
Full raw build and harness logs are retained. No log line is filtered.

All six builds completed, created the actual installed operator libraries and
executed in the full-workload public and leaf/composition checks. Actual file
hashes and source signatures are in the build artifact manifest.

1. CMake says CMAKE_CROSS_PLATFORM_COMPILER is unused. The selected native
   build produces linux/x86_64 host libraries on its x86_64 runtime, so the
   supplied cross-platform-only variable does not affect this native build.
   No cross-platform or other-architecture acceptance is inferred.
2. CANN ge_common headers emit deprecation notices for ge_error_codes.h and
   ge_api_types.h. Their messages state planned removal after2027-06; current
   declarations compile and link successfully. They concern vendor host API
   includes, not kernel arithmetic, layout, or synchronization. No vendor
   source was changed to hide these notices.
3. CANN's ascendc_impl_build.py:194 emits a Python3.12 SyntaxWarning for an
   invalid escape sequence in a template. Fresh tokenization of the actual
   generated build helper locates the outer string at154:11..211:3; the nested
   registration text contains a raw regex for ASCENDC_API_VERSION. This warning
   is attributed to the outer Python string literal, not the nested regex.
   The fresh template SHA256 is
   ec9f69a86e2d3a9da17dedab13716fa1ffad80140ea88a74cbdb8ead035ec385.
   The generated registration helper compiles all six selected operators;
   actual loading/execution is verified by the full workload. No suppressed
   warning, changed regex or alternate backend is used.

Fresh extraction found zero vector-loop-condition warnings in these builds.
Lowered event balance is empty for all three new entries, and full native
NaN-poisoned independent leaves/composition returned finite matching outputs.
The above findings do not stand in for the remaining grouped/mode/bd grid or
same-card timing; those acceptance stages remain separate.
