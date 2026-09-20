# Per-launch layout cost

Measured on dev-B, bd4. Each row is one actual layout launch. Event acquisition
and clean end-to-end timing are separate. No internal per-launch synchronization
is used in event acquisition. Source signatures and every non-layout event are
retained in the original attribution JSON files.

| Version | T | Call | Layout index | Operator | Input bytes | Output bytes | Device event ms | Host dispatch ms |
|---|---:|---|---:|---|---:|---:|---:|---:|
| original-64 | 1024 | plain_forward | 1 | KdaLayoutBf16Bf16Kernel | 8388608 | 8388608 | 9.686538696 | 0.334773213 |
| original-64 | 1024 | plain_forward | 2 | KdaLayoutBf16Bf16Kernel | 8388608 | 8388608 | 10.133864403 | 0.031718053 |
| original-64 | 1024 | plain_forward | 3 | KdaLayoutBf16Bf16Kernel | 8388608 | 8388608 | 9.947971344 | 0.026860042 |
| original-64 | 1024 | plain_forward | 4 | KdaLayoutF32F32Kernel | 16777216 | 16777216 | 7.972678185 | 0.027701026 |
| original-64 | 1024 | plain_forward | 5 | KdaLayoutF32F32Kernel | 131072 | 131072 | 0.077753998 | 0.024526846 |
| original-64 | 1024 | plain_forward | 6 | KdaLayoutBf16Bf16Kernel | 8388608 | 8388608 | 7.873418808 | 0.028272858 |
| original-64 | 1024 | cached_forward | 1 | KdaLayoutBf16Bf16Kernel | 8388608 | 8388608 | 9.956005096 | 0.051086070 |
| original-64 | 1024 | cached_forward | 2 | KdaLayoutBf16Bf16Kernel | 8388608 | 8388608 | 10.251111984 | 0.029824907 |
| original-64 | 1024 | cached_forward | 3 | KdaLayoutBf16Bf16Kernel | 8388608 | 8388608 | 10.336807251 | 0.027620932 |
| original-64 | 1024 | cached_forward | 4 | KdaLayoutF32F32Kernel | 16777216 | 16777216 | 8.063847542 | 0.031257048 |
| original-64 | 1024 | cached_forward | 5 | KdaLayoutF32F32Kernel | 131072 | 131072 | 0.077239998 | 0.025948975 |
| original-64 | 1024 | cached_forward | 6 | KdaLayoutF32Bf16Kernel | 16777216 | 8388608 | 8.214047432 | 0.540520996 |
| original-64 | 1024 | cached_forward | 7 | KdaLayoutBf16Bf16Kernel | 4194304 | 4194304 | 4.987085819 | 0.028923154 |
| original-64 | 1024 | cached_forward | 8 | KdaLayoutBf16Bf16Kernel | 4194304 | 4194304 | 5.249082088 | 0.024806941 |
| original-64 | 1024 | cached_forward | 9 | KdaLayoutBf16Bf16Kernel | 8388608 | 8388608 | 10.385175705 | 0.024846988 |
| original-64 | 1024 | cached_forward | 10 | KdaLayoutBf16Bf16Kernel | 8388608 | 8388608 | 10.066954613 | 0.023795059 |
| original-64 | 1024 | cached_forward | 11 | KdaLayoutBf16Bf16Kernel | 8388608 | 8388608 | 10.110802650 | 0.025307992 |
| original-64 | 1024 | cached_forward | 12 | KdaLayoutBf16Bf16Kernel | 8388608 | 8388608 | 10.168863297 | 0.024206005 |
| original-64 | 1024 | cached_forward | 13 | KdaLayoutBf16Bf16Kernel | 8388608 | 8388608 | 9.823128700 | 0.023925910 |
| original-64 | 1024 | cached_forward | 14 | KdaLayoutBf16Bf16Kernel | 8388608 | 8388608 | 10.129195213 | 0.024036039 |
| original-64 | 1024 | backward_existing_caches | 1 | KdaLayoutBf16Bf16Kernel | 7872512 | 131072 | 0.135060996 | 0.033519929 |
| original-64 | 4096 | plain_forward | 1 | KdaLayoutBf16Bf16Kernel | 33554432 | 33554432 | 39.851516724 | 0.061902916 |
| original-64 | 4096 | plain_forward | 2 | KdaLayoutBf16Bf16Kernel | 33554432 | 33554432 | 39.846176147 | 0.029823976 |
| original-64 | 4096 | plain_forward | 3 | KdaLayoutBf16Bf16Kernel | 33554432 | 33554432 | 39.625591278 | 0.025688903 |
| original-64 | 4096 | plain_forward | 4 | KdaLayoutF32F32Kernel | 67108864 | 67108864 | 39.788162231 | 0.037206104 |
| original-64 | 4096 | plain_forward | 5 | KdaLayoutF32F32Kernel | 524288 | 524288 | 0.288814008 | 0.025878893 |
| original-64 | 4096 | plain_forward | 6 | KdaLayoutBf16Bf16Kernel | 33554432 | 33554432 | 37.715950012 | 0.028052134 |
| original-64 | 4096 | cached_forward | 1 | KdaLayoutBf16Bf16Kernel | 33554432 | 33554432 | 39.664047241 | 0.092789065 |
| original-64 | 4096 | cached_forward | 2 | KdaLayoutBf16Bf16Kernel | 33554432 | 33554432 | 39.971702576 | 0.043264823 |
| original-64 | 4096 | cached_forward | 3 | KdaLayoutBf16Bf16Kernel | 33554432 | 33554432 | 39.654720306 | 0.026680063 |
| original-64 | 4096 | cached_forward | 4 | KdaLayoutF32F32Kernel | 67108864 | 67108864 | 39.694889069 | 0.037085963 |
| original-64 | 4096 | cached_forward | 5 | KdaLayoutF32F32Kernel | 524288 | 524288 | 0.287396014 | 0.025497982 |
| original-64 | 4096 | cached_forward | 6 | KdaLayoutF32Bf16Kernel | 67108864 | 33554432 | 32.809719086 | 2.616256010 |
| original-64 | 4096 | cached_forward | 7 | KdaLayoutBf16Bf16Kernel | 16777216 | 16777216 | 19.840572357 | 0.029925024 |
| original-64 | 4096 | cached_forward | 8 | KdaLayoutBf16Bf16Kernel | 16777216 | 16777216 | 19.869863510 | 0.024536857 |
| original-64 | 4096 | cached_forward | 9 | KdaLayoutBf16Bf16Kernel | 33554432 | 33554432 | 39.516822815 | 0.025037909 |
| original-64 | 4096 | cached_forward | 10 | KdaLayoutBf16Bf16Kernel | 33554432 | 33554432 | 39.761844635 | 0.024396926 |
| original-64 | 4096 | cached_forward | 11 | KdaLayoutBf16Bf16Kernel | 33554432 | 33554432 | 39.492908478 | 0.023255125 |
| original-64 | 4096 | cached_forward | 12 | KdaLayoutBf16Bf16Kernel | 33554432 | 33554432 | 39.809211731 | 0.023214146 |
| original-64 | 4096 | cached_forward | 13 | KdaLayoutBf16Bf16Kernel | 33554432 | 33554432 | 39.621757507 | 0.024125911 |
| original-64 | 4096 | cached_forward | 14 | KdaLayoutBf16Bf16Kernel | 33554432 | 33554432 | 39.610370636 | 0.023144064 |
| original-64 | 4096 | backward_existing_caches | 1 | KdaLayoutBf16Bf16Kernel | 33038336 | 524288 | 0.637149990 | 0.030835858 |
| batched-4096 | 1024 | plain_forward | 1 | KdaLayoutBf16Bf16Kernel | 8388608 | 8388608 | 0.265861988 | 0.178457936 |
| batched-4096 | 1024 | plain_forward | 2 | KdaLayoutBf16Bf16Kernel | 8388608 | 8388608 | 0.257090002 | 0.029544113 |
| batched-4096 | 1024 | plain_forward | 3 | KdaLayoutBf16Bf16Kernel | 8388608 | 8388608 | 0.249252006 | 0.025719171 |
| batched-4096 | 1024 | plain_forward | 4 | KdaLayoutF32F32Kernel | 16777216 | 16777216 | 0.237984002 | 0.026148977 |
| batched-4096 | 1024 | plain_forward | 5 | KdaLayoutF32F32Kernel | 131072 | 131072 | 0.006968000 | 0.024296110 |
| batched-4096 | 1024 | plain_forward | 6 | KdaLayoutBf16Bf16Kernel | 8388608 | 8388608 | 0.205044001 | 0.026448863 |
| batched-4096 | 1024 | cached_forward | 1 | KdaLayoutBf16Bf16Kernel | 8388608 | 8388608 | 0.278932005 | 0.043435022 |
| batched-4096 | 1024 | cached_forward | 2 | KdaLayoutBf16Bf16Kernel | 8388608 | 8388608 | 0.247947007 | 0.029894989 |
| batched-4096 | 1024 | cached_forward | 3 | KdaLayoutBf16Bf16Kernel | 8388608 | 8388608 | 0.250579000 | 0.026078895 |
| batched-4096 | 1024 | cached_forward | 4 | KdaLayoutF32F32Kernel | 16777216 | 16777216 | 0.238845006 | 0.025657937 |
| batched-4096 | 1024 | cached_forward | 5 | KdaLayoutF32F32Kernel | 131072 | 131072 | 0.006808000 | 0.024857931 |
| batched-4096 | 1024 | cached_forward | 6 | KdaLayoutF32Bf16Kernel | 16777216 | 8388608 | 0.536652029 | 0.287421048 |
| batched-4096 | 1024 | cached_forward | 7 | KdaLayoutBf16Bf16Kernel | 4194304 | 4194304 | 0.131439999 | 0.026840018 |
| batched-4096 | 1024 | cached_forward | 8 | KdaLayoutBf16Bf16Kernel | 4194304 | 4194304 | 0.129050002 | 0.024275854 |
| batched-4096 | 1024 | cached_forward | 9 | KdaLayoutBf16Bf16Kernel | 8388608 | 8388608 | 0.248401001 | 0.024075853 |
| batched-4096 | 1024 | cached_forward | 10 | KdaLayoutBf16Bf16Kernel | 8388608 | 8388608 | 0.244448006 | 0.023476081 |
| batched-4096 | 1024 | cached_forward | 11 | KdaLayoutBf16Bf16Kernel | 8388608 | 8388608 | 0.243440002 | 0.023365021 |
| batched-4096 | 1024 | cached_forward | 12 | KdaLayoutBf16Bf16Kernel | 8388608 | 8388608 | 0.253760993 | 0.023975968 |
| batched-4096 | 1024 | cached_forward | 13 | KdaLayoutBf16Bf16Kernel | 8388608 | 8388608 | 0.238837004 | 0.024036039 |
| batched-4096 | 1024 | cached_forward | 14 | KdaLayoutBf16Bf16Kernel | 8388608 | 8388608 | 0.244058996 | 0.023073982 |
| batched-4096 | 1024 | backward_existing_caches | 1 | KdaLayoutBf16Bf16Kernel | 7872512 | 131072 | 0.039733998 | 0.031097094 |
| batched-4096 | 4096 | plain_forward | 1 | KdaLayoutBf16Bf16Kernel | 33554432 | 33554432 | 0.989241004 | 0.037026126 |
| batched-4096 | 4096 | plain_forward | 2 | KdaLayoutBf16Bf16Kernel | 33554432 | 33554432 | 0.964749992 | 0.027410919 |
| batched-4096 | 4096 | plain_forward | 3 | KdaLayoutBf16Bf16Kernel | 33554432 | 33554432 | 0.966992021 | 0.025017885 |
| batched-4096 | 4096 | plain_forward | 4 | KdaLayoutF32F32Kernel | 67108864 | 67108864 | 1.054594994 | 0.025288202 |
| batched-4096 | 4096 | plain_forward | 5 | KdaLayoutF32F32Kernel | 524288 | 524288 | 0.059316002 | 0.024106121 |
| batched-4096 | 4096 | plain_forward | 6 | KdaLayoutBf16Bf16Kernel | 33554432 | 33554432 | 0.962202013 | 0.024967827 |
| batched-4096 | 4096 | cached_forward | 1 | KdaLayoutBf16Bf16Kernel | 33554432 | 33554432 | 0.987149000 | 0.045287888 |
| batched-4096 | 4096 | cached_forward | 2 | KdaLayoutBf16Bf16Kernel | 33554432 | 33554432 | 0.966547012 | 0.028823037 |
| batched-4096 | 4096 | cached_forward | 3 | KdaLayoutBf16Bf16Kernel | 33554432 | 33554432 | 0.964358985 | 0.028112205 |
| batched-4096 | 4096 | cached_forward | 4 | KdaLayoutF32F32Kernel | 67108864 | 67108864 | 1.055024028 | 0.025598099 |
| batched-4096 | 4096 | cached_forward | 5 | KdaLayoutF32F32Kernel | 524288 | 524288 | 0.059850998 | 0.024016015 |
| batched-4096 | 4096 | cached_forward | 6 | KdaLayoutF32Bf16Kernel | 67108864 | 33554432 | 2.164746046 | 0.983966980 |
| batched-4096 | 4096 | cached_forward | 7 | KdaLayoutBf16Bf16Kernel | 16777216 | 16777216 | 0.507516026 | 0.028041890 |
| batched-4096 | 4096 | cached_forward | 8 | KdaLayoutBf16Bf16Kernel | 16777216 | 16777216 | 0.502397001 | 0.024436042 |
| batched-4096 | 4096 | cached_forward | 9 | KdaLayoutBf16Bf16Kernel | 33554432 | 33554432 | 0.963886023 | 0.023254892 |
| batched-4096 | 4096 | cached_forward | 10 | KdaLayoutBf16Bf16Kernel | 33554432 | 33554432 | 0.959982991 | 0.023504952 |
| batched-4096 | 4096 | cached_forward | 11 | KdaLayoutBf16Bf16Kernel | 33554432 | 33554432 | 0.964484990 | 0.023245113 |
| batched-4096 | 4096 | cached_forward | 12 | KdaLayoutBf16Bf16Kernel | 33554432 | 33554432 | 0.964358985 | 0.022704015 |
| batched-4096 | 4096 | cached_forward | 13 | KdaLayoutBf16Bf16Kernel | 33554432 | 33554432 | 0.962984979 | 0.024146168 |
| batched-4096 | 4096 | cached_forward | 14 | KdaLayoutBf16Bf16Kernel | 33554432 | 33554432 | 0.961335003 | 0.022683991 |
| batched-4096 | 4096 | backward_existing_caches | 1 | KdaLayoutBf16Bf16Kernel | 33038336 | 524288 | 0.042188998 | 0.033680117 |
