import time

import pygame
import cv2
import mediapipe as mp

from headtracking import HeadTracker
from feedback import feedBackEngine
from observation import ObservationEngine
from gaze_zones import ZONE_CATALOG, GazeZoneClassifier, HEAD_SCALE_FRAMES
from scenegen import SceneGen, auth_client
from scenes import Scene, Metrics
from UI import UI
from api_client import ApiClient
from session_recorder import SessionRecorder


scene = SceneGen(1920, 1080, 60)
ui = UI(scene.screen, scene.W, scene.H)
scene.ui = ui
scene.selected_scene = "left_lane_change"
scene.state = "home"

api_client = ApiClient(auth_client)
backend_session_id = None
recorder = None

scene_manager = None
metrics = None
cap = None
tracker = None
feedback = None


prev_smoothed = None
prev_angles = None
prev_prev_angles = None
baseline_angles = None
baseline_buffer = []
BASELINE_FRAMES = 60
prev_rel = None
prev_prev_rel = None
gaze_calibrated = False
gaze_warmup_frames = 30
gaze_warmup_count = 0
gaze_phase = "center_ref"

gaze_forward_buffer = []
zone_anchor_index = 0
zone_anchor_buffer = []
ZONE_ANCHOR_FRAMES = 30
head_scale_buffer = []

gaze_zone_classifier = GazeZoneClassifier()
observation_engine = ObservationEngine(gaze_zone_classifier)

DEADZONE_PITCH = 2
DEADZONE_YAW = 1
DEADZONE_ROLL = 1


def apply_deadzone(angle, threshold):
    distance = max(0, abs(angle) - threshold)
    sign = 1 if angle >= 0 else -1
    return sign * distance


def clamp(x, upper):
    return max(-upper, min(upper, x))


def angle_diff_deg(a, b):
    return (a - b + 180) % 360 - 180


running = True
simulation_initialized = False

while running:

    if scene.state not in {"simulation", "calibration"}:
        if simulation_initialized:
            if cap is not None:
                cap.release()
                cap = None
            cv2.destroyAllWindows()
            simulation_initialized = False

        running = scene.update()
        continue

    if scene.state in {"simulation", "calibration"} and not simulation_initialized:
        scene_manager = Scene(scene.selected_scene)
        metrics = Metrics(scene_manager.current_scene.expected_sequence)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("cannot access camera")
            break

        tracker = HeadTracker()
        feedback = feedBackEngine()
        tracker.reset_gaze()

        prev_smoothed = None
        prev_angles = None
        prev_prev_angles = None
        baseline_angles = None
        baseline_buffer = []
        prev_rel = None
        prev_prev_rel = None
        simulation_initialized = True
        gaze_calibrated = False
        gaze_warmup_frames = 30
        gaze_warmup_count = 0
        gaze_phase = "center_ref"
        gaze_forward_buffer = []
        zone_anchor_index = 0
        zone_anchor_buffer = []
        head_scale_buffer = []
        gaze_zone_classifier.clear()
        observation_engine.reset()
        

    ret, frame = cap.read()
    if not ret:
        break

    results = tracker.process_frame(frame)

    if not results.multi_face_landmarks:
        if prev_angles and prev_prev_angles:
            final_pitch = prev_angles["pitch"] + (prev_angles["pitch"] - prev_prev_angles["pitch"])
            final_yaw = prev_angles["yaw"] + (prev_angles["yaw"] - prev_prev_angles["yaw"])
            final_roll = prev_angles["roll"] + (prev_angles["roll"] - prev_prev_angles["roll"])
        elif prev_angles:
            final_pitch = prev_angles["pitch"]
            final_yaw = prev_angles["yaw"]
            final_roll = prev_angles["roll"]
        else:
            final_pitch, final_yaw, final_roll = 0, 0, 0

        final_pitch = apply_deadzone(final_pitch, DEADZONE_PITCH)
        final_yaw = apply_deadzone(final_yaw, DEADZONE_YAW)
        final_roll = apply_deadzone(final_roll, DEADZONE_ROLL)

        pose = feedback.update(final_pitch, final_yaw, final_roll)
        observed_zone = observation_engine.update(pose, None, None)
        if recorder is not None:
            recorder.on_pose(pose, final_yaw, final_pitch, time.time())
        progress_data = scene_manager.get_progress_data()

        running = scene.update(final_pitch, final_yaw, final_roll, observed_zone, progress_data)
        if not running:
            break

        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        continue

    for face_landmarks in results.multi_face_landmarks:
        tracker.mp_drawing.draw_landmarks(
            frame,
            face_landmarks,
            mp.solutions.face_mesh.FACEMESH_TESSELATION
        )

        raw_pos = tracker.get_body_pos(face_landmarks)
        smoothed_pos = tracker.smoothed_points(raw_pos, prev_smoothed, 0.2)

        vectors = tracker.pitch_vectors(smoothed_pos)
        pitch = vectors["pitch_angle"]
        yaw = vectors["yaw_angle"]
        roll = vectors["roll_angle"]

        if baseline_angles is not None:
            rel_pitch = angle_diff_deg(pitch, baseline_angles["pitch"])
            rel_yaw = angle_diff_deg(yaw, baseline_angles["yaw"])
            rel_roll = angle_diff_deg(roll, baseline_angles["roll"])
        else:
            rel_pitch = 0.0
            rel_yaw = 0.0
            rel_roll = 0.0

        if scene.state == "calibration" :
            if baseline_angles is None:
                baseline_buffer.append((pitch, yaw, roll))

                if len(baseline_buffer) >= 5:
                    recent = baseline_buffer[-5:]
                    spread_yaw = max(y for _, y, _ in recent) - min(y for _, y, _ in recent)
                    if spread_yaw > 4:
                        baseline_buffer.clear()
                        prev_smoothed = smoothed_pos
                        continue
                
                
                head_progress = min(1.0, len(baseline_buffer) / BASELINE_FRAMES)
                calibration_progress_data = {
                    "progress": 0.6 * head_progress, 
                    "status_text": "Hold still and face forward"
                }

                running = scene.update(0,0,0, "FORWARD", calibration_progress_data)
                if not running :
                    break


                if len(baseline_buffer) < BASELINE_FRAMES:
                    
                    prev_smoothed = smoothed_pos
                    continue

                avg_pitch = sum(p for p, _, _ in baseline_buffer) / BASELINE_FRAMES
                avg_yaw = sum(y for _, y, _ in baseline_buffer) / BASELINE_FRAMES
                avg_roll = sum(r for _, _, r in baseline_buffer) / BASELINE_FRAMES

                baseline_angles = {
                    "pitch": avg_pitch,
                    "yaw": avg_yaw,
                    "roll": avg_roll
                }

                tracker.reset_gaze()
                gaze_calibrated = False
                gaze_warmup_count = 0

                prev_angles = {"pitch": 0, "yaw": 0, "roll": 0}
                prev_prev_angles = None
                prev_rel = {"pitch": 0, "yaw": 0, "roll": 0}
                prev_prev_rel = None
                prev_smoothed = smoothed_pos

                continue

            if not gaze_calibrated : 

                if gaze_warmup_count < gaze_warmup_frames : 
                    gaze_warmup_count += 1
                    
                    
                    gaze_progress = min(1.0, gaze_warmup_count / gaze_warmup_frames)
                    calibration_progress_data = {
                        "progress": 0.6 + 0.4 * gaze_progress, 
                        "status_text": "Look directly at the center dot"
                    } 
                    
                    running = scene.update(0,0,0, "FORWARD", calibration_progress_data, "center")

                    if not running :
                        break

                    prev_smoothed = smoothed_pos
                    continue
                    
                    
                norm_x, norm_y = tracker.normalized_gaze(face_landmarks)
                
                eye_data = tracker.get_gaze_pos(face_landmarks)
                left_height = tracker.compute_eye_height(eye_data["left_eye"])
                right_height = tracker.compute_eye_height(eye_data["right_eye"])

                
                display_target = "center"

                world_x, world_y = gaze_zone_classifier.to_world_gaze(norm_x, norm_y, rel_yaw, rel_pitch)

                if gaze_phase == "center_ref" :
                    done = tracker.collect_gaze_ref("center_ref", left_height, right_height )
                    gaze_forward_buffer.append((world_x, world_y))
                    print("CENTER_REF COUNT", len(tracker.eye_height_buffer["left"]))

                    display_target = "center"

                    calibration_progress_data = {
                        "progress": 0.68,
                        "status_text": "Look directly at the center point"
                    }

                    if done :
                        gaze_ref_calibrated = tracker.finalize_center_ref()
                        if gaze_ref_calibrated :
                            if gaze_forward_buffer :
                                gaze_zone_classifier.set_forward_baseline(list(gaze_forward_buffer))
                                gaze_forward_buffer.clear()
                            if ZONE_CATALOG :
                                gaze_phase = "zone_anchor"
                                zone_anchor_index = 0
                                zone_anchor_buffer = []
                            else :
                                gaze_zone_classifier.finalize_anchors()
                                gaze_phase = "head_scale"
                                head_scale_buffer = []

                elif gaze_phase == "zone_anchor" :
                    zone = ZONE_CATALOG[zone_anchor_index]
                    display_target = zone.name

                    zone_anchor_buffer.append((world_x, world_y))

                    zone_progress = (zone_anchor_index + len(zone_anchor_buffer) / ZONE_ANCHOR_FRAMES) / max(1, len(ZONE_CATALOG))
                    calibration_progress_data = {
                        "progress": 0.75 + 0.2 * zone_progress,
                        "status_text": zone.calibration_prompt,
                    }

                    if len(zone_anchor_buffer) >= ZONE_ANCHOR_FRAMES :
                        gaze_zone_classifier.register_anchor(zone.name, list(zone_anchor_buffer))
                        zone_anchor_buffer = []
                        zone_anchor_index += 1
                        if zone_anchor_index >= len(ZONE_CATALOG) :
                            gaze_zone_classifier.finalize_anchors()
                            gaze_phase = "head_scale"
                            head_scale_buffer = []

                elif gaze_phase == "head_scale" :
                    display_target = "center"
                    head_scale_buffer.append((norm_x, norm_y, rel_yaw, rel_pitch))

                    hs_progress = len(head_scale_buffer) / HEAD_SCALE_FRAMES
                    calibration_progress_data = {
                        "progress": 0.95 + 0.05 * hs_progress,
                        "status_text": "Keep your eyes on the dot and slowly move your head around",
                    }

                    if len(head_scale_buffer) >= HEAD_SCALE_FRAMES :
                        gaze_zone_classifier.calibrate_head_scale(list(head_scale_buffer))
                        head_scale_buffer = []
                        gaze_calibrated = True
                        scene.state = "simulation"
                        scene.start_fade_in()

                running = scene.update(0,0,0, "FORWARD", calibration_progress_data, display_target)

                if not running :
                    break

                prev_smoothed = smoothed_pos

                print("GAZE PHASE:", gaze_phase)
                continue


        norm_x, norm_y = tracker.normalized_gaze(face_landmarks)

        final_pitch, final_yaw, final_roll = rel_pitch, rel_yaw, rel_roll

        final_pitch = apply_deadzone(final_pitch, DEADZONE_PITCH)
        final_yaw = apply_deadzone(final_yaw, DEADZONE_YAW)
        final_roll = apply_deadzone(final_roll, DEADZONE_ROLL)

        pose = feedback.update(final_pitch, final_yaw, final_roll)
        world_x, world_y = gaze_zone_classifier.to_world_gaze(norm_x, norm_y, rel_yaw, rel_pitch)
        observed_zone = observation_engine.update(
            pose,
            gaze_x=world_x, gaze_y=world_y,
            norm_x=norm_x, norm_y=norm_y,
        )
        eye_ratio = gaze_zone_classifier.eye_contribution_ratio(norm_x, norm_y, pose) if gaze_zone_classifier.has_anchor(pose) else None
        print("[obs] head=", pose, "world_gaze=", (world_x, world_y),
              "eye_ratio=", eye_ratio, "zone=", observed_zone,
              "held=", observation_engine.zone_counter)
        if recorder is not None:
            recorder.on_pose(pose, final_yaw, final_pitch, time.time())
        progress_data = scene_manager.get_progress_data()

        running = scene.update(final_pitch, final_yaw, final_roll, observed_zone, progress_data)
        if not running:
            break

        prev_prev_rel = prev_rel.copy() if prev_rel else None
        prev_rel = {"pitch": rel_pitch, "yaw": rel_yaw, "roll": rel_roll}

        prev_prev_angles = prev_angles.copy() if prev_angles else None
        prev_angles = {"pitch": final_pitch, "yaw": final_yaw, "roll": final_roll}

        prev_smoothed = smoothed_pos

        outcome = scene_manager.evaluation(observation_engine.zone_counter, observed_zone)

        if outcome and outcome["finished"] :
            result = outcome["result"]
            score = metrics.sequence_score(result)

            if recorder is not None:
                recorder.finalize(final_yaw, final_pitch, time.time())
                api_client.complete_session(backend_session_id, score, result)
                backend_session_id = None
                recorder = None

            scene.last_score = score
            scene.last_result = result
            scene.state = "results"
            scene.start_fade_in()

            continue

    cv2.imshow("Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

if cap is not None:
    cap.release()

cv2.destroyAllWindows()
pygame.quit()