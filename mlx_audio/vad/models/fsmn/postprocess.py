"""
FSMN-VAD 后处理状态机

从 FunASR fsmn_vad_streaming/model.py 移植，去掉 torch 依赖。
所有 tensor 操作改为 numpy / 纯 Python。
"""

import math
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class VadStateMachine(Enum):
    kVadInStateStartPointNotDetected = 1
    kVadInStateInSpeechSegment = 2
    kVadInStateEndPointDetected = 3


class FrameState(Enum):
    kFrameStateInvalid = -1
    kFrameStateSpeech = 1
    kFrameStateSil = 0


class AudioChangeState(Enum):
    kChangeStateSpeech2Speech = 0
    kChangeStateSpeech2Sil = 1
    kChangeStateSil2Sil = 2
    kChangeStateSil2Speech = 3
    kChangeStateNoBegin = 4
    kChangeStateInvalid = 5


class VadDetectMode(Enum):
    kVadSingleUtteranceDetectMode = 0
    kVadMutipleUtteranceDetectMode = 1


class VADXOptions:
    def __init__(
        self,
        sample_rate=16000,
        detect_mode=1,
        snr_mode=0,
        max_end_silence_time=800,
        max_start_silence_time=3000,
        do_start_point_detection=True,
        do_end_point_detection=True,
        window_size_ms=200,
        sil_to_speech_time_thres=150,
        speech_to_sil_time_thres=150,
        speech_2_noise_ratio=1.0,
        do_extend=1,
        lookback_time_start_point=200,
        lookahead_time_end_point=100,
        max_single_segment_time=60000,
        nn_eval_block_size=8,
        dcd_block_size=4,
        snr_thres=-100.0,
        noise_frame_num_used_for_snr=100,
        decibel_thres=-100.0,
        speech_noise_thres=0.6,
        fe_prior_thres=1e-4,
        silence_pdf_num=1,
        sil_pdf_ids=None,
        speech_noise_thresh_low=-0.1,
        speech_noise_thresh_high=0.3,
        output_frame_probs=False,
        frame_in_ms=10,
        frame_length_ms=25,
        **kwargs,
    ):
        self.sample_rate = sample_rate
        self.detect_mode = detect_mode
        self.snr_mode = snr_mode
        self.max_end_silence_time = max_end_silence_time
        self.max_start_silence_time = max_start_silence_time
        self.do_start_point_detection = do_start_point_detection
        self.do_end_point_detection = do_end_point_detection
        self.window_size_ms = window_size_ms
        self.sil_to_speech_time_thres = sil_to_speech_time_thres
        self.speech_to_sil_time_thres = speech_to_sil_time_thres
        self.speech_2_noise_ratio = speech_2_noise_ratio
        self.do_extend = do_extend
        self.lookback_time_start_point = lookback_time_start_point
        self.lookahead_time_end_point = lookahead_time_end_point
        self.max_single_segment_time = max_single_segment_time
        self.nn_eval_block_size = nn_eval_block_size
        self.dcd_block_size = dcd_block_size
        self.snr_thres = snr_thres
        self.noise_frame_num_used_for_snr = noise_frame_num_used_for_snr
        self.decibel_thres = decibel_thres
        self.speech_noise_thres = speech_noise_thres
        self.fe_prior_thres = fe_prior_thres
        self.silence_pdf_num = silence_pdf_num
        self.sil_pdf_ids = sil_pdf_ids if sil_pdf_ids is not None else [0]
        self.speech_noise_thresh_low = speech_noise_thresh_low
        self.speech_noise_thresh_high = speech_noise_thresh_high
        self.output_frame_probs = output_frame_probs
        self.frame_in_ms = frame_in_ms
        self.frame_length_ms = frame_length_ms


class E2EVadSpeechBufWithDoa:
    def __init__(self):
        self.Reset()

    def Reset(self):
        self.start_ms = 0
        self.end_ms = 0
        self.buffer = []
        self.contain_seg_start_point = False
        self.contain_seg_end_point = False
        self.doa = 0


class E2EVadFrameProb:
    def __init__(self):
        self.noise_prob = 0.0
        self.speech_prob = 0.0
        self.score = 0.0
        self.frame_id = 0
        self.frm_state = 0


class WindowDetector:
    def __init__(
        self, window_size_ms, sil_to_speech_time, speech_to_sil_time, frame_size_ms
    ):
        self.window_size_ms = window_size_ms
        self.frame_size_ms = frame_size_ms
        self.win_size_frame = int(window_size_ms / frame_size_ms)
        self.sil_to_speech_frmcnt_thres = int(sil_to_speech_time / frame_size_ms)
        self.speech_to_sil_frmcnt_thres = int(speech_to_sil_time / frame_size_ms)
        self.Reset()

    def Reset(self):
        self.cur_win_pos = 0
        self.win_sum = 0
        self.win_state = [0] * self.win_size_frame
        self.pre_frame_state = FrameState.kFrameStateSil
        self.cur_frame_state = FrameState.kFrameStateSil
        self.voice_last_frame_count = 0
        self.noise_last_frame_count = 0
        self.hydre_frame_count = 0

    def GetWinSize(self):
        return self.win_size_frame

    def DetectOneFrame(self, frameState, frame_count, cache={}):
        cur_frame_state = 1 if frameState == FrameState.kFrameStateSpeech else 0
        self.win_sum -= self.win_state[self.cur_win_pos]
        self.win_sum += cur_frame_state
        self.win_state[self.cur_win_pos] = cur_frame_state
        self.cur_win_pos = (self.cur_win_pos + 1) % self.win_size_frame

        if (
            self.pre_frame_state == FrameState.kFrameStateSil
            and self.win_sum >= self.sil_to_speech_frmcnt_thres
        ):
            self.pre_frame_state = FrameState.kFrameStateSpeech
            return AudioChangeState.kChangeStateSil2Speech
        if (
            self.pre_frame_state == FrameState.kFrameStateSpeech
            and self.win_sum <= self.speech_to_sil_frmcnt_thres
        ):
            self.pre_frame_state = FrameState.kFrameStateSil
            return AudioChangeState.kChangeStateSpeech2Sil
        if self.pre_frame_state == FrameState.kFrameStateSil:
            return AudioChangeState.kChangeStateSil2Sil
        if self.pre_frame_state == FrameState.kFrameStateSpeech:
            return AudioChangeState.kChangeStateSpeech2Speech
        return AudioChangeState.kChangeStateInvalid


class Stats:
    def __init__(self, sil_pdf_ids, max_end_sil_frame_cnt_thresh, speech_noise_thres):
        self.data_buf_start_frame = 0
        self.frm_cnt = 0
        self.latest_confirmed_speech_frame = 0
        self.lastest_confirmed_silence_frame = -1
        self.continous_silence_frame_count = 0
        self.vad_state_machine = VadStateMachine.kVadInStateStartPointNotDetected
        self.confirmed_start_frame = -1
        self.confirmed_end_frame = -1
        self.number_end_time_detected = 0
        self.sil_frame = 0
        self.sil_pdf_ids = sil_pdf_ids
        self.noise_average_decibel = -100.0
        self.pre_end_silence_detected = False
        self.next_seg = True
        self.output_data_buf = []
        self.output_data_buf_offset = 0
        self.frame_probs = []
        self.max_end_sil_frame_cnt_thresh = max_end_sil_frame_cnt_thresh
        self.speech_noise_thres = speech_noise_thres
        self.scores = None
        self.max_time_out = False
        self.decibel = []
        self.data_buf = None
        self.data_buf_all = None
        self.waveform = None
        self.last_drop_frames = 0


class VADPostProcess:
    """VAD 后处理: 帧级 scores + waveform → 语音段时间戳."""

    def __init__(self, opts: VADXOptions):
        self.vad_opts = opts

    def init_cache(self):
        cache = {}
        cache["windows_detector"] = WindowDetector(
            self.vad_opts.window_size_ms,
            self.vad_opts.sil_to_speech_time_thres,
            self.vad_opts.speech_to_sil_time_thres,
            self.vad_opts.frame_in_ms,
        )
        cache["stats"] = Stats(
            sil_pdf_ids=self.vad_opts.sil_pdf_ids,
            max_end_sil_frame_cnt_thresh=(
                self.vad_opts.max_end_silence_time
                - self.vad_opts.speech_to_sil_time_thres
            ),
            speech_noise_thres=self.vad_opts.speech_noise_thres,
        )
        return cache

    def compute_decibel(self, waveform: np.ndarray, cache: dict):
        """计算每帧分贝值. waveform: [samples] float32."""
        sr = self.vad_opts.sample_rate
        frame_sample_length = int(self.vad_opts.frame_length_ms * sr / 1000)
        frame_shift_length = int(self.vad_opts.frame_in_ms * sr / 1000)

        if cache["stats"].data_buf_all is None:
            cache["stats"].data_buf_all = waveform.copy()
            cache["stats"].data_buf = cache["stats"].data_buf_all
        else:
            cache["stats"].data_buf_all = np.concatenate(
                [cache["stats"].data_buf_all, waveform]
            )

        offsets = np.arange(
            0, len(waveform) - frame_sample_length + 1, frame_shift_length
        )
        if len(offsets) == 0:
            return
        frames = waveform[offsets[:, np.newaxis] + np.arange(frame_sample_length)]
        decibel = 10 * np.log10(np.sum(np.square(frames), axis=1) + 1e-6)
        cache["stats"].decibel.extend(decibel.tolist())

    def compute_scores(self, scores: np.ndarray, cache: dict):
        """累积 encoder 输出的 scores. scores: [1, T, D] numpy."""
        num_frames = scores.shape[1]
        self.vad_opts.nn_eval_block_size = num_frames
        cache["stats"].frm_cnt += num_frames
        if cache["stats"].scores is None:
            cache["stats"].scores = scores
        else:
            cache["stats"].scores = np.concatenate(
                [cache["stats"].scores, scores], axis=1
            )

    def LatencyFrmNumAtStartPoint(self, cache):
        vad_latency = cache["windows_detector"].GetWinSize()
        if self.vad_opts.do_extend:
            vad_latency += int(
                self.vad_opts.lookback_time_start_point / self.vad_opts.frame_in_ms
            )
        return vad_latency

    def PopDataBufTillFrame(self, frame_idx, cache):
        while cache["stats"].data_buf_start_frame < frame_idx:
            if len(cache["stats"].data_buf) >= int(
                self.vad_opts.frame_in_ms * self.vad_opts.sample_rate / 1000
            ):
                cache["stats"].data_buf_start_frame += 1
                cache["stats"].data_buf = cache["stats"].data_buf_all[
                    (
                        cache["stats"].data_buf_start_frame
                        - cache["stats"].last_drop_frames
                    )
                    * int(
                        self.vad_opts.frame_in_ms * self.vad_opts.sample_rate / 1000
                    ) :
                ]
            else:
                break

    def PopDataToOutputBuf(
        self,
        start_frm,
        frm_cnt,
        first_frm_is_start_point,
        last_frm_is_end_point,
        end_point_is_sent_end,
        cache,
    ):
        self.PopDataBufTillFrame(start_frm, cache)
        expected_sample_number = int(
            frm_cnt * self.vad_opts.sample_rate * self.vad_opts.frame_in_ms / 1000
        )
        if last_frm_is_end_point:
            extra_sample = max(
                0,
                int(
                    self.vad_opts.frame_length_ms * self.vad_opts.sample_rate / 1000
                    - self.vad_opts.sample_rate * self.vad_opts.frame_in_ms / 1000
                ),
            )
            expected_sample_number += extra_sample
        if end_point_is_sent_end:
            expected_sample_number = max(
                expected_sample_number, len(cache["stats"].data_buf)
            )

        if len(cache["stats"].output_data_buf) == 0 or first_frm_is_start_point:
            cache["stats"].output_data_buf.append(E2EVadSpeechBufWithDoa())
            cache["stats"].output_data_buf[-1].Reset()
            cache["stats"].output_data_buf[-1].start_ms = (
                start_frm * self.vad_opts.frame_in_ms
            )
            cache["stats"].output_data_buf[-1].end_ms = (
                cache["stats"].output_data_buf[-1].start_ms
            )
            cache["stats"].output_data_buf[-1].doa = 0
        cur_seg = cache["stats"].output_data_buf[-1]
        cache["stats"].data_buf_start_frame += frm_cnt
        cur_seg.end_ms = (start_frm + frm_cnt) * self.vad_opts.frame_in_ms
        if first_frm_is_start_point:
            cur_seg.contain_seg_start_point = True
        if last_frm_is_end_point:
            cur_seg.contain_seg_end_point = True

    def OnSilenceDetected(self, valid_frame, cache):
        cache["stats"].lastest_confirmed_silence_frame = valid_frame
        if (
            cache["stats"].vad_state_machine
            == VadStateMachine.kVadInStateStartPointNotDetected
        ):
            self.PopDataBufTillFrame(valid_frame, cache)

    def OnVoiceDetected(self, valid_frame, cache):
        cache["stats"].latest_confirmed_speech_frame = valid_frame
        self.PopDataToOutputBuf(valid_frame, 1, False, False, False, cache)

    def OnVoiceStart(self, start_frame, fake_result=False, cache=None):
        if cache["stats"].confirmed_start_frame != -1:
            pass
        else:
            cache["stats"].confirmed_start_frame = start_frame
        if (
            not fake_result
            and cache["stats"].vad_state_machine
            == VadStateMachine.kVadInStateStartPointNotDetected
        ):
            self.PopDataToOutputBuf(
                cache["stats"].confirmed_start_frame, 1, True, False, False, cache
            )

    def OnVoiceEnd(self, end_frame, fake_result, is_last_frame, cache):
        for t in range(cache["stats"].latest_confirmed_speech_frame + 1, end_frame):
            self.OnVoiceDetected(t, cache)
        if cache["stats"].confirmed_end_frame != -1:
            pass
        else:
            cache["stats"].confirmed_end_frame = end_frame
        if not fake_result:
            cache["stats"].sil_frame = 0
            self.PopDataToOutputBuf(
                cache["stats"].confirmed_end_frame, 1, False, True, is_last_frame, cache
            )
        cache["stats"].number_end_time_detected += 1

    def MaybeOnVoiceEndIfLastFrame(self, is_final_frame, cur_frm_idx, cache):
        if is_final_frame:
            self.OnVoiceEnd(cur_frm_idx, False, True, cache)
            cache["stats"].vad_state_machine = (
                VadStateMachine.kVadInStateEndPointDetected
            )

    def ResetDetection(self, cache):
        cache["stats"].continous_silence_frame_count = 0
        cache["stats"].latest_confirmed_speech_frame = 0
        cache["stats"].lastest_confirmed_silence_frame = -1
        cache["stats"].confirmed_start_frame = -1
        cache["stats"].confirmed_end_frame = -1
        cache["stats"].vad_state_machine = (
            VadStateMachine.kVadInStateStartPointNotDetected
        )
        cache["windows_detector"].Reset()
        cache["stats"].sil_frame = 0
        cache["stats"].frame_probs = []
        if cache["stats"].output_data_buf:
            assert cache["stats"].output_data_buf[-1].contain_seg_end_point
            drop_frames = int(
                cache["stats"].output_data_buf[-1].end_ms / self.vad_opts.frame_in_ms
            )
            real_drop_frames = drop_frames - cache["stats"].last_drop_frames
            cache["stats"].last_drop_frames = drop_frames
            cache["stats"].data_buf_all = cache["stats"].data_buf_all[
                real_drop_frames
                * int(self.vad_opts.frame_in_ms * self.vad_opts.sample_rate / 1000) :
            ]
            cache["stats"].decibel = cache["stats"].decibel[real_drop_frames:]
            cache["stats"].scores = cache["stats"].scores[:, real_drop_frames:, :]

    def GetFrameState(self, t, cache):
        frame_state = FrameState.kFrameStateInvalid
        # Safe boundary check
        if t < 0 or t >= len(cache["stats"].decibel):
            return FrameState.kFrameStateSil
        cur_decibel = cache["stats"].decibel[t]
        cur_snr = cur_decibel - cache["stats"].noise_average_decibel
        if cur_decibel < self.vad_opts.decibel_thres:
            return FrameState.kFrameStateSil

        sum_score = 0.0
        if len(cache["stats"].sil_pdf_ids) > 0:
            if len(cache["stats"].sil_pdf_ids) > 1:
                sum_score = sum(
                    float(cache["stats"].scores[0][t][sid])
                    for sid in cache["stats"].sil_pdf_ids
                )
            else:
                sum_score = float(
                    cache["stats"].scores[0][t][cache["stats"].sil_pdf_ids[0]]
                )
            sum_score = max(min(sum_score, 1.0 - 1e-7), 1e-7)
            noise_prob = math.log(sum_score) * self.vad_opts.speech_2_noise_ratio
            sum_score = 1.0 - sum_score

        speech_prob = math.log(sum_score)
        if self.vad_opts.output_frame_probs:
            fp = E2EVadFrameProb()
            fp.noise_prob = noise_prob
            fp.speech_prob = speech_prob
            fp.score = sum_score
            fp.frame_id = t
            cache["stats"].frame_probs.append(fp)

        if (
            math.exp(speech_prob)
            >= math.exp(noise_prob) + cache["stats"].speech_noise_thres
        ):
            if (
                cur_snr >= self.vad_opts.snr_thres
                and cur_decibel >= self.vad_opts.decibel_thres
            ):
                frame_state = FrameState.kFrameStateSpeech
            else:
                frame_state = FrameState.kFrameStateSil
        else:
            frame_state = FrameState.kFrameStateSil
            if cache["stats"].noise_average_decibel < -99.9:
                cache["stats"].noise_average_decibel = cur_decibel
            else:
                cache["stats"].noise_average_decibel = (
                    cur_decibel
                    + cache["stats"].noise_average_decibel
                    * (self.vad_opts.noise_frame_num_used_for_snr - 1)
                ) / self.vad_opts.noise_frame_num_used_for_snr

        return frame_state

    def DetectOneFrame(self, cur_frm_state, cur_frm_idx, is_final_frame, cache):
        tmp_cur_frm_state = FrameState.kFrameStateInvalid
        if cur_frm_state == FrameState.kFrameStateSpeech:
            tmp_cur_frm_state = (
                FrameState.kFrameStateSpeech
                if math.fabs(1.0) > self.vad_opts.fe_prior_thres
                else FrameState.kFrameStateSil
            )
        elif cur_frm_state == FrameState.kFrameStateSil:
            tmp_cur_frm_state = FrameState.kFrameStateSil

        state_change = cache["windows_detector"].DetectOneFrame(
            tmp_cur_frm_state, cur_frm_idx, cache
        )
        frm_shift_in_ms = self.vad_opts.frame_in_ms

        if state_change == AudioChangeState.kChangeStateSil2Speech:
            cache["stats"].continous_silence_frame_count = 0
            cache["stats"].pre_end_silence_detected = False
            if (
                cache["stats"].vad_state_machine
                == VadStateMachine.kVadInStateStartPointNotDetected
            ):
                start_frame = max(
                    cache["stats"].data_buf_start_frame,
                    cur_frm_idx - self.LatencyFrmNumAtStartPoint(cache),
                )
                self.OnVoiceStart(start_frame, cache=cache)
                cache["stats"].vad_state_machine = (
                    VadStateMachine.kVadInStateInSpeechSegment
                )
                for t in range(start_frame + 1, cur_frm_idx + 1):
                    self.OnVoiceDetected(t, cache)
            elif (
                cache["stats"].vad_state_machine
                == VadStateMachine.kVadInStateInSpeechSegment
            ):
                for t in range(
                    cache["stats"].latest_confirmed_speech_frame + 1, cur_frm_idx
                ):
                    self.OnVoiceDetected(t, cache)
                if (
                    cur_frm_idx - cache["stats"].confirmed_start_frame + 1
                    > self.vad_opts.max_single_segment_time / frm_shift_in_ms
                ):
                    self.OnVoiceEnd(cur_frm_idx, False, False, cache)
                    cache["stats"].vad_state_machine = (
                        VadStateMachine.kVadInStateEndPointDetected
                    )
                elif not is_final_frame:
                    self.OnVoiceDetected(cur_frm_idx, cache)
                else:
                    self.MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx, cache)

        elif state_change == AudioChangeState.kChangeStateSpeech2Sil:
            cache["stats"].continous_silence_frame_count = 0
            if (
                cache["stats"].vad_state_machine
                == VadStateMachine.kVadInStateInSpeechSegment
            ):
                if (
                    cur_frm_idx - cache["stats"].confirmed_start_frame + 1
                    > self.vad_opts.max_single_segment_time / frm_shift_in_ms
                ):
                    self.OnVoiceEnd(cur_frm_idx, False, False, cache)
                    cache["stats"].vad_state_machine = (
                        VadStateMachine.kVadInStateEndPointDetected
                    )
                elif not is_final_frame:
                    self.OnVoiceDetected(cur_frm_idx, cache)
                else:
                    self.MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx, cache)

        elif state_change == AudioChangeState.kChangeStateSpeech2Speech:
            cache["stats"].continous_silence_frame_count = 0
            if (
                cache["stats"].vad_state_machine
                == VadStateMachine.kVadInStateInSpeechSegment
            ):
                if (
                    cur_frm_idx - cache["stats"].confirmed_start_frame + 1
                    > self.vad_opts.max_single_segment_time / frm_shift_in_ms
                ):
                    cache["stats"].max_time_out = True
                    self.OnVoiceEnd(cur_frm_idx, False, False, cache)
                    cache["stats"].vad_state_machine = (
                        VadStateMachine.kVadInStateEndPointDetected
                    )
                elif not is_final_frame:
                    self.OnVoiceDetected(cur_frm_idx, cache)
                else:
                    self.MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx, cache)

        elif state_change == AudioChangeState.kChangeStateSil2Sil:
            cache["stats"].continous_silence_frame_count += 1
            if (
                cache["stats"].vad_state_machine
                == VadStateMachine.kVadInStateStartPointNotDetected
            ):
                if (
                    self.vad_opts.detect_mode
                    == VadDetectMode.kVadSingleUtteranceDetectMode.value
                    and cache["stats"].continous_silence_frame_count * frm_shift_in_ms
                    > self.vad_opts.max_start_silence_time
                ) or (is_final_frame and cache["stats"].number_end_time_detected == 0):
                    for t in range(
                        cache["stats"].lastest_confirmed_silence_frame + 1, cur_frm_idx
                    ):
                        self.OnSilenceDetected(t, cache)
                    self.OnVoiceStart(0, True, cache)
                    self.OnVoiceEnd(0, True, False, cache)
                    cache["stats"].vad_state_machine = (
                        VadStateMachine.kVadInStateEndPointDetected
                    )
                else:
                    if cur_frm_idx >= self.LatencyFrmNumAtStartPoint(cache):
                        self.OnSilenceDetected(
                            cur_frm_idx - self.LatencyFrmNumAtStartPoint(cache), cache
                        )
            elif (
                cache["stats"].vad_state_machine
                == VadStateMachine.kVadInStateInSpeechSegment
            ):
                if (
                    cache["stats"].continous_silence_frame_count * frm_shift_in_ms
                    >= cache["stats"].max_end_sil_frame_cnt_thresh
                ):
                    lookback_frame = int(
                        cache["stats"].max_end_sil_frame_cnt_thresh / frm_shift_in_ms
                    )
                    if self.vad_opts.do_extend:
                        lookback_frame -= int(
                            self.vad_opts.lookahead_time_end_point / frm_shift_in_ms
                        )
                        lookback_frame -= 1
                        lookback_frame = max(0, lookback_frame)
                    self.OnVoiceEnd(cur_frm_idx - lookback_frame, False, False, cache)
                    cache["stats"].vad_state_machine = (
                        VadStateMachine.kVadInStateEndPointDetected
                    )
                elif (
                    cur_frm_idx - cache["stats"].confirmed_start_frame + 1
                    > self.vad_opts.max_single_segment_time / frm_shift_in_ms
                ):
                    self.OnVoiceEnd(cur_frm_idx, False, False, cache)
                    cache["stats"].vad_state_machine = (
                        VadStateMachine.kVadInStateEndPointDetected
                    )
                elif self.vad_opts.do_extend and not is_final_frame:
                    if cache["stats"].continous_silence_frame_count <= int(
                        self.vad_opts.lookahead_time_end_point / frm_shift_in_ms
                    ):
                        self.OnVoiceDetected(cur_frm_idx, cache)
                else:
                    self.MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx, cache)

        if (
            cache["stats"].vad_state_machine
            == VadStateMachine.kVadInStateEndPointDetected
            and self.vad_opts.detect_mode
            == VadDetectMode.kVadMutipleUtteranceDetectMode.value
        ):
            self.ResetDetection(cache)

    def DetectCommonFrames(self, cache):
        if (
            cache["stats"].vad_state_machine
            == VadStateMachine.kVadInStateEndPointDetected
        ):
            return
        for i in range(self.vad_opts.nn_eval_block_size - 1, -1, -1):
            frame_state = self.GetFrameState(
                cache["stats"].frm_cnt - 1 - i - cache["stats"].last_drop_frames, cache
            )
            self.DetectOneFrame(
                frame_state, cache["stats"].frm_cnt - 1 - i, False, cache
            )

    def DetectLastFrames(self, cache):
        if (
            cache["stats"].vad_state_machine
            == VadStateMachine.kVadInStateEndPointDetected
        ):
            return
        for i in range(self.vad_opts.nn_eval_block_size - 1, -1, -1):
            frame_state = self.GetFrameState(
                cache["stats"].frm_cnt - 1 - i - cache["stats"].last_drop_frames, cache
            )
            if i != 0:
                self.DetectOneFrame(
                    frame_state, cache["stats"].frm_cnt - 1 - i, False, cache
                )
            else:
                self.DetectOneFrame(
                    frame_state, cache["stats"].frm_cnt - 1, True, cache
                )

    def forward(
        self,
        scores: np.ndarray,
        waveform: np.ndarray,
        cache: dict,
        is_final: bool = True,
    ) -> List[List[int]]:
        """
        处理一个 chunk 的 scores 和 waveform.

        Args:
            scores: [1, T, D] numpy (encoder softmax 输出)
            waveform: [samples] float32 numpy (原始波形)
            cache: 状态缓存
            is_final: 是否为最后一个 chunk

        Returns:
            segments: [[start_ms, end_ms], ...]
        """
        cache["stats"].waveform = waveform
        self.compute_decibel(waveform, cache)
        self.compute_scores(scores, cache)

        if not is_final:
            self.DetectCommonFrames(cache)
        else:
            self.DetectLastFrames(cache)

        segments = []
        if len(cache["stats"].output_data_buf) > 0:
            for i in range(
                cache["stats"].output_data_buf_offset,
                len(cache["stats"].output_data_buf),
            ):
                if not is_final and (
                    not cache["stats"].output_data_buf[i].contain_seg_start_point
                    or not cache["stats"].output_data_buf[i].contain_seg_end_point
                ):
                    continue
                segment = [
                    cache["stats"].output_data_buf[i].start_ms,
                    cache["stats"].output_data_buf[i].end_ms,
                ]
                cache["stats"].output_data_buf_offset += 1
                segments.append(segment)

        return segments
