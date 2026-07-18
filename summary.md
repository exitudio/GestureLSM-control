# Control Eval Comparison: 26-07-15 12:01:49 to 22:40:47

Inclusive run range:

- Start: `26-07-15_12-01-49__absolute_Control_den1-5-100`
- End: `26-07-15_22-40-47__absolute_Control_den1-5-100_opt1-10-100`

Shared settings omitted from the table:

- `densities`: `[1, 5, 100] per chunk`
- `control settings`: `15`
- `available samples`: `15`
- `max_samples total`: `15`
- `scale`: `1.0`
- `weight`: `1.0`
- `active_norm`: `sqrt`
- `freeze_root`: `False`

| Run | space | iters_early | iters_late | post_iters | fgd | align | l1div | traj_err_5cm | loc_err_5cm | avg_err_cm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `26-07-15_12-01-49__absolute_Control_den1-5-100` | absolute | 1 | 30 | N/A | 0.572071 | 0.782265 | 12.091880 | 0.693227 | 0.713927 | 9.148263 |
| `26-07-15_13-28-19__relative_tto` | relative | 0 | 0 | 30 | 0.469240 | 0.767106 | 12.095418 | 0.087649 | 0.091613 | 2.072365 |
| `26-07-15_13-28-50__absolute_tto` | absolute | 0 | 0 | 30 | 0.486208 | 0.789570 | 12.229258 | 0.900398 | 0.789492 | 11.991972 |
| `26-07-15_13-44-22__relative_tto100` | relative | 0 | 0 | 100 | 0.473551 | 0.768349 | 12.218925 | 0.003984 | 0.000070 | 0.494496 |
| `26-07-15_13-44-48__absolute_tto100` | absolute | 0 | 0 | 100 | 0.509278 | 0.790440 | 12.659519 | 0.247012 | 0.193553 | 3.171630 |
| `26-07-15_22-40-47__absolute_Control_den1-5-100_opt1-10-100` | absolute | 1 | 10 | 100 | 0.591042 | 0.790457 | 12.212195 | 0.119522 | 0.077030 | 1.801452 |

Notes:

- Lower is better for `fgd`, `traj_err_5cm`, `loc_err_5cm`, and `avg_err_cm`.
- Higher is typically better for `align` and `l1div`, depending on the evaluation target.
- `post_iters` is `N/A` when the setting is absent from the log.


# Control Eval Comparison: 26-07-17 01:41:12 to 13:20:29

Inclusive run range:

- Start: `26-07-17_01-41-12__relative_den1-2-5_opt1-5-30_NormLinear_RERUN`
- End: `26-07-17_13-20-29__relative_den1-2-5_opt0-5-30_NormLinear_delay1+fullDec_StepIncrease_fixPostInWave_start100`

Shared settings omitted from the table:

- `space`: `relative`
- `densities`: `[1, 2, 5] per chunk`
- `control settings`: `15`
- `available samples`: `15`
- `max_samples total`: `15`
- `scale`: `1.0`
- `weight`: `1.0`
- `active_norm`: `linear`
- `freeze_root`: `False`

<table>
  <thead>
    <tr>
      <th>mark</th>
      <th>Run</th>
      <th>chunk_delay</th>
      <th>schedule</th>
      <th>post behavior</th>
      <th>iters_early</th>
      <th>iters_late</th>
      <th>late_start</th>
      <th>post_iters</th>
      <th>avg_err_cm</th>
      <th>fgd</th>
      <th>align</th>
      <th>l1div</th>
      <th>foot_skating</th>
      <th>traj_err_5cm</th>
      <th>loc_err_5cm</th>
      <th>gen_time_s</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color:#fff3cd;">
      <td><strong>BASELINE</strong></td>
      <td><strong><code>26-07-17_01-41-12__relative_den1-2-5_opt1-5-30_NormLinear_RERUN</code></strong></td>
      <td>0</td><td>hard switch</td><td>final post after denoise</td><td>1</td><td>5</td><td>300</td><td>30</td><td>1.196631</td><td>0.438180</td><td>0.748048</td><td>11.848854</td><td>0.066894</td><td>0.043825</td><td>0.046602</td><td>N/A</td>
    </tr>
    <tr>
      <td></td>
      <td><code>26-07-17_01-48-36__relative_den1-2-5_opt0-30-30_NormLinear_delay1+fullDec</code></td>
      <td>1</td><td>linear, shared min timestep</td><td>final post after denoise</td><td>0</td><td>30</td><td>300</td><td>30</td><td>incomplete</td><td>incomplete</td><td>incomplete</td><td>incomplete</td><td>incomplete</td><td>incomplete</td><td>incomplete</td><td>incomplete</td>
    </tr>
    <tr>
      <td></td>
      <td><code>26-07-17_01-49-15__relative_den1-2-5_opt0-30-30_NormLinear_delay0_ttoIncrease</code></td>
      <td>0</td><td>linear, shared min timestep</td><td>final post after denoise</td><td>0</td><td>30</td><td>300</td><td>30</td><td>0.333379</td><td>0.444727</td><td>0.754878</td><td>11.711279</td><td>0.065636</td><td>0.007968</td><td>0.001942</td><td>N/A</td>
    </tr>
    <tr>
      <td></td>
      <td><code>26-07-17_04-16-50__relative_den1-2-5_opt0-5-30_NormLinear_delay1+fullDec_RERUN</code></td>
      <td>1</td><td>hard switch</td><td>final post after denoise</td><td>0</td><td>5</td><td>300</td><td>30</td><td>0.801991</td><td>0.430187</td><td>0.753571</td><td>11.793847</td><td>0.071526</td><td>0.035857</td><td>0.023301</td><td>2941</td>
    </tr>
    <tr>
      <td></td>
      <td><code>26-07-17_04-29-14__relative_den1-2-5_opt0-10-30_NormLinear_delay1+fullDec_StepIncrease</code></td>
      <td>1</td><td>linear, shared min timestep</td><td>final post after denoise</td><td>0</td><td>10</td><td>300</td><td>30</td><td>0.458950</td><td>0.434513</td><td>0.758269</td><td>11.807413</td><td>0.070306</td><td>0.011952</td><td>0.003883</td><td>4757</td>
    </tr>
    <tr style="background-color:#d1ecf1;">
      <td><strong>POST-IN-WAVE</strong></td>
      <td><strong><code>26-07-17_07-04-55__relative_den1-2-5_opt0-10-30_NormLinear_delay1+fullDec_StepIncrease_fixPostInWave</code></strong></td>
      <td>1</td><td>linear, per-chunk iteration mask</td><td>post runs inside wave</td><td>0</td><td>10</td><td>300</td><td>30</td><td>2.640418</td><td>0.431137</td><td>0.755345</td><td>11.497326</td><td>0.073598</td><td>0.243028</td><td>0.123301</td><td>4995</td>
    </tr>
    <tr>
      <td></td>
      <td><code>26-07-17_13-08-13__relative_den1-2-5_opt0-10-30_NormLinear_delay0+fullDec_StepIncrease_fixPostInWave</code></td>
      <td>0</td><td>linear, per-chunk iteration mask</td><td>final post after denoise</td><td>0</td><td>10</td><td>300</td><td>30</td><td>0.904310</td><td>0.438458</td><td>0.752251</td><td>11.806503</td><td>0.066473</td><td>0.031873</td><td>0.032039</td><td>1455</td>
    </tr>
    <tr>
      <td></td>
      <td><code>26-07-17_13-18-06__relative_den1-2-5_opt0-5-30_NormLinear_delay1+fullDec_StepIncrease_fixPostInWave</code></td>
      <td>1</td><td>linear, per-chunk iteration mask</td><td>post runs inside wave</td><td>0</td><td>5</td><td>300</td><td>30</td><td>2.925428</td><td>0.425886</td><td>0.753514</td><td>11.594867</td><td>0.074713</td><td>0.290837</td><td>0.144660</td><td>2819</td>
    </tr>
    <tr>
      <td></td>
      <td><code>26-07-17_13-20-29__relative_den1-2-5_opt0-5-30_NormLinear_delay1+fullDec_StepIncrease_fixPostInWave_start100</code></td>
      <td>1</td><td>linear, per-chunk iteration mask</td><td>post runs inside wave</td><td>0</td><td>5</td><td>100</td><td>30</td><td>2.898324</td><td>0.425797</td><td>0.754645</td><td>11.658201</td><td>0.075312</td><td>0.274900</td><td>0.142718</td><td>2664</td>
    </tr>
  </tbody>
</table>

Notes:

- Highlighted rows are the requested baseline (`01-41-12`) and selected post-in-wave comparison (`07-04-55`).
- `01-48-36` was not completed; its log stops at `sample_index=11`, so aggregate metrics are unavailable.
- `01-41-12` and `04-16-50` used a hard switch: `iters_early` before `late_start`, then `iters_late` after `late_start`.
- `01-49-15`, `01-48-36`, and `04-29-14` used linear step increase, but the schedule was based on a shared `t.min()` for the active wave, not per chunk.
- `07-04-55` and the `13-*` delayed runs used the newer per-chunk iteration mask and moved post optimization into the delayed wave schedule.
- `13-08-13` used the newer code with `chunk_delay=0`, so post optimization still ran as a final post-denoise block.
- Best control accuracy in this range by `avg_err_cm` is still `01-49-15` at `0.333379 cm`, followed by `04-29-14` at `0.458950 cm`.
- The post-in-wave run `07-04-55` had similar `fgd` to the best delayed runs, but much worse trajectory/local control error in this run.

# Control Eval Comparison: 26-07-18 02:41:08 to 05:48:54

Inclusive run range:

- Start: `26-07-18_02-41-08__absolute_den1-2-5_opt0-5-30_NormLinear_delay0+fullDec_StepIncrease_start100`
- End: `26-07-18_05-48-54__absolute_den1-2-5_opt0-30-30_NormLinear_delay0+fullDec_StepIncrease_start100`

Shared settings omitted from the table:

- `space`: `absolute`
- `densities`: `[1, 2, 5] per chunk`
- `iters_early`: `0`
- `late_start`: `100`
- `post_iters`: `30`
- `scale`: `1.0`
- `weight`: `1.0`
- `active_norm`: `linear`
- `freeze_root`: `False`

| Run | chunk_delay | iters_late | avg_err_cm | traj_5 | traj_10 | traj_20 | traj_50 | loc_5 | loc_10 | loc_20 | loc_50 | fgd | align | l1div | foot_skating | gen_time_s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `26-07-18_02-41-08__absolute_den1-2-5_opt0-5-30_NormLinear_delay0+fullDec_StepIncrease_start100` | 0 | 5 | 7.169152 | 0.625498 | N/A | N/A | N/A | 0.534951 | N/A | N/A | N/A | 0.444298 | 0.761429 | 12.089540 | 0.054155 | 758 |
| `26-07-18_02-57-07__absolute_den1-2-5_opt0-5-30_NormLinear_delay1+fullDec_StepIncrease_start100` | 1 | 5 | 9.541210 | 0.828685 | N/A | N/A | N/A | 0.686408 | N/A | N/A | N/A | 0.439104 | 0.765565 | 11.634272 | 0.078068 | 2407 |
| `26-07-18_02-57-43__absolute_den1-2-5_opt0-10-30_NormLinear_delay1+fullDec_StepIncrease_start100` | 1 | 10 | 8.946033 | 0.828685 | N/A | N/A | N/A | 0.660194 | N/A | N/A | N/A | 0.452479 | 0.770137 | 11.436724 | 0.078684 | 4266 |
| `26-07-18_03-44-16__absolute_den1-2-5_opt0-5-30_NormLinear_delay1+fullDec_StepIncrease_start100` | 1 | 5 | 9.429157 | 0.832669 | 0.486056 | 0.195219 | 0.011952 | 0.691262 | 0.320388 | 0.093204 | 0.002913 | 0.440867 | 0.762931 | 11.648256 | 0.072473 | 2421 |
| `26-07-18_04-34-19__absolute_den1-2-5_opt0-20-30_NormLinear_delay1+fullDec_StepIncrease_start100` | 1 | 20 | 7.841073 | 0.737052 | 0.342629 | 0.127490 | 0.015936 | 0.586408 | 0.231068 | 0.054369 | 0.004854 | 0.461893 | 0.777664 | 11.313787 | 0.098244 | 7949 |
| `26-07-18_05-48-25__absolute_den1-2-5_opt0-20-30_NormLinear_delay0+fullDec_StepIncrease_start100` | 0 | 20 | 6.698243 | 0.669323 | 0.310757 | 0.043825 | 0.003984 | 0.547573 | 0.199029 | 0.021359 | 0.001942 | 0.449436 | 0.760542 | 11.962539 | 0.054501 | 1411 |
| `26-07-18_05-48-54__absolute_den1-2-5_opt0-30-30_NormLinear_delay0+fullDec_StepIncrease_start100` | 0 | 30 | 5.651711 | 0.549801 | 0.227092 | 0.027888 | 0.000000 | 0.427184 | 0.142718 | 0.014563 | 0.000000 | 0.451415 | 0.760436 | 11.939730 | 0.057758 | 1744 |

Notes:

- Lower is better for `avg_err_cm`, `traj_*`, `loc_*`, `fgd`, and `foot_skating`.
- Higher is typically better for `align` and `l1div`, depending on the target tradeoff.
- `N/A` means the run finished before the 10/20/50cm error metrics were added to logging.
- Best `avg_err_cm` in this group is `05-48-54` at `5.651711 cm`; best `fgd` is `02-57-07` at `0.439104`.
- The `chunk_delay=0` runs are much faster here than delayed wave runs, but absolute control errors remain high compared with the relative-space table above.
