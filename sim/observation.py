from gaze_zones import MIN_EYE_CONTRIBUTION_RATIO

DEFAULT_STABILITY_FRAMES = 5
GAZE_EMA_ALPHA = 0.4


class ObservationEngine:

    def __init__(self, classifier, stability_frames=DEFAULT_STABILITY_FRAMES):
        self.classifier = classifier
        self.stability_frames = stability_frames

        self.last_zone = "FORWARD"
        self.zone_counter = 0
        self.confirmed_zone = "FORWARD"

        self._smoothed_gaze = None

    def reset(self):
        self.last_zone = "FORWARD"
        self.zone_counter = 0
        self.confirmed_zone = "FORWARD"
        self._smoothed_gaze = None

    def estimate_zone(self, head_pose, gaze_x=None, gaze_y=None, norm_x=None, norm_y=None):
        gaze_zone = self.classifier.classify(gaze_x, gaze_y)

        if head_pose == "FORWARD":
            return gaze_zone if gaze_zone is not None else "FORWARD"

        if not self.classifier.has_anchor(head_pose):
            return head_pose

        if gaze_zone == head_pose:
            ratio = self.classifier.eye_contribution_ratio(norm_x, norm_y, head_pose)
            if ratio >= MIN_EYE_CONTRIBUTION_RATIO:
                return head_pose
            return "FORWARD"

        if self.classifier.is_at_forward_baseline(gaze_x, gaze_y):
            return "FORWARD"

        return head_pose

    def update(self, head_pose, gaze_x=None, gaze_y=None, norm_x=None, norm_y=None):
        if gaze_x is None or gaze_y is None:
            smoothed_x, smoothed_y = None, None
        else:
            if self._smoothed_gaze is None:
                self._smoothed_gaze = (gaze_x, gaze_y)
            else:
                sx, sy = self._smoothed_gaze
                self._smoothed_gaze = (
                    sx + GAZE_EMA_ALPHA * (gaze_x - sx),
                    sy + GAZE_EMA_ALPHA * (gaze_y - sy),
                )
            smoothed_x, smoothed_y = self._smoothed_gaze

        zone = self.estimate_zone(head_pose, smoothed_x, smoothed_y, norm_x, norm_y)

        if zone == self.last_zone:
            self.zone_counter += 1
        else:
            self.zone_counter = 1
        self.last_zone = zone

        if self.zone_counter >= self.stability_frames:
            prev_confirmed = self.confirmed_zone
            self.confirmed_zone = zone
            if zone != prev_confirmed:
                print("[obs] confirmed zone:", prev_confirmed, "->", zone)

        return self.confirmed_zone
