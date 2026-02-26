# This is a sample Python script.
import os
import json
import numpy as np
from tqdm import tqdm
from pynput.keyboard import Key, Controller

try:
    import cv2
except ImportError as e:
    cv2 = None
    _cv2_import_error = e

try:
    import tensorflow as tf
except ImportError:
    tf = None



# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    os.system("screencapture screen.png")
    from PIL import Image
    image = Image.open('screen.png')


def save_kilter_image(path, number=None):
    # Take Screenshot
    os.system("screencapture screen.png")
    # Save one Image for Text Recognition and one for the board image
    from PIL import Image
    image_text = Image.open('screen.png')
    image_kilter = Image.open('screen.png')
    # Crop the desired Image parts outs - still handcrafted
    image_text = image_text.crop((1300, 300, 1750, 525))
    image_kilter = image_kilter.crop((980, 520, 2040, 1725))
    # text = pytesseract.image_to_string(image_text)
    image_text.save(path + 'kilter_image' + str(number) + 'label_.png')
    image_kilter.save(path + 'kilter_image' + str(number) + '.png')
    keyboard = Controller()
    keyboard.press(Key.right)
    keyboard.release(Key.right)


def create_labels_from_text(path):
    folder = path + '/Labels/'
    for filename in tqdm(os.listdir(path)):
        if filename.endswith(('_.png')):  # Check if file is an image
            from PIL import Image
            import pytesseract
            img = Image.open(os.path.join(path, filename))
            string_label = pytesseract.image_to_string(img)
            #print(filename)
            #print(string_label)
            string_lines = string_label.split('\n')
            #print(len(string_lines))
            if (len(string_lines) == 1):
                pass
            elif('V' in string_lines[1]):
                string_vrating = string_lines[1].split('V')
                string_vrating_only = string_vrating[1].split(' ')
                string_vrating_only = string_vrating_only[0]
                if string_vrating_only == 'O':
                    string_vrating_only = '0'
                #print(string_vrating_only)
                one_hot_vector = np.zeros(17)
                one_hot_vector[int(string_vrating_only)] = 1
                np.save(folder + filename[:-5] + 'Numpy', one_hot_vector)
                with open(folder + filename[:-5] + '.txt', 'w') as f:
                    f.write(string_vrating_only)

def load_Kilter_Dataset(path, n_images = 1000, image_dim = (1165, 1060, 4)):
    images = np.zeros(((n_images,) + image_dim))
    labels = np.zeros((n_images,17))
    file_names = []
    i = 0
    for filename in tqdm(os.listdir(path)):
        if filename.endswith(('.png')):  # Check if file is an image
            matching_label = filename[:-4] + 'labelNumpy.npy'
            label_path = os.path.join((path + '/Labels/'), matching_label)
            if os.path.exists(label_path):
                file_names.append(filename)
                from PIL import Image
                img = Image.open(os.path.join(path, filename))
                matching_label = np.load(path + '/Labels/' + filename[:-4] + 'labelNumpy.npy')
                images[i] = img
                labels[i] = matching_label
                i+=1
                if i >= n_images:
                    break

    return images, labels, file_names


def _require_cv2():
    if cv2 is None:
        raise ImportError(
            "OpenCV (cv2) is required for hold/ring detection. "
            "Install with: pip install opencv-python"
        ) from _cv2_import_error


def build_hold_map(
    empty_board_path,
    output_json,
    min_area=50,
    max_area=12000,
    row_gap_scale=1.5,
    col_gap_scale=1.5,
    use_bolt_holes=True,
    min_cluster_size=4,
    expected_rows=None,
    expected_cols=None,
    row_step_px=None,
    col_step_px=None,
    shape_match_radius=30,
):
    """
    Detects holds from an empty board image and builds a row/col grid.
    Saves JSON with: rows, cols, holds[{id,x,y,row,col,area}].
    """
    _require_cv2()
    img = cv2.imread(empty_board_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {empty_board_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) Primary detection: hold silhouettes
    holds = []
    _, hold_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(hold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        holds.append({"x": cx, "y": cy, "area_shape": float(area)})

    # 2) Optional refinement: bolt holes inside silhouettes
    bolt_holes = []
    if use_bolt_holes:
        # Detect small dark bolt holes to get stable hold centers.
        inv = cv2.bitwise_not(gray)
        inv = cv2.GaussianBlur(inv, (5, 5), 0)
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 255
        params.filterByArea = True
        params.minArea = 8
        params.maxArea = 260
        params.filterByCircularity = True
        params.minCircularity = 0.25
        params.filterByInertia = True
        params.minInertiaRatio = 0.2
        params.filterByConvexity = False
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(inv)
        for kp in keypoints:
            bolt_holes.append((float(kp.pt[0]), float(kp.pt[1])))

    if holds and bolt_holes:
        bolt_holes = np.array(bolt_holes, dtype=np.float32)
        for h in holds:
            cx, cy = float(h["x"]), float(h["y"])
            dists = np.sum((bolt_holes - np.array([cx, cy])) ** 2, axis=1)
            inside = bolt_holes[dists <= shape_match_radius ** 2]
            if len(inside) > 0:
                # Use average bolt position as center (handles 1 or 2 bolts)
                h["x"] = int(np.mean(inside[:, 0]))
                h["y"] = int(np.mean(inside[:, 1]))

    if not holds:
        raise RuntimeError("No holds detected. Check threshold or area limits.")

    xs = np.array([h["x"] for h in holds], dtype=np.float32)
    ys = np.array([h["y"] for h in holds], dtype=np.float32)

    def _cluster_1d(values, gap_scale, min_cluster_size):
        values_sorted = np.sort(values)
        diffs = np.diff(values_sorted)
        if len(diffs) == 0:
            return [float(values_sorted[0])]
        gap = np.percentile(diffs, 75) * gap_scale
        clusters = [[values_sorted[0]]]
        for v_prev, v in zip(values_sorted[:-1], values_sorted[1:]):
            if (v - v_prev) > gap:
                clusters.append([v])
            else:
                clusters[-1].append(v)
        centers = [float(np.mean(c)) for c in clusters]
        counts = [len(c) for c in clusters]

        # Merge tiny clusters into nearest larger cluster to avoid extra rows/cols.
        for i, count in enumerate(counts):
            if count >= min_cluster_size:
                continue
            if len(centers) <= 1:
                continue
            nearest = None
            nearest_dist = None
            for j, c in enumerate(centers):
                if i == j:
                    continue
                dist = abs(centers[i] - c)
                if nearest is None or dist < nearest_dist:
                    nearest = j
                    nearest_dist = dist
            if nearest is not None:
                clusters[nearest].extend(clusters[i])
                clusters[i] = []

        centers = [float(np.mean(c)) for c in clusters if len(c) > 0]
        return centers

    def _cluster_fixed_step(values, step_px):
        values_sorted = np.sort(values)
        clusters = []
        current = [values_sorted[0]]
        current_center = values_sorted[0]
        for v in values_sorted[1:]:
            if abs(v - current_center) <= step_px:
                current.append(v)
                current_center = float(np.mean(current))
            else:
                clusters.append(current)
                current = [v]
                current_center = v
        clusters.append(current)
        centers = [float(np.mean(c)) for c in clusters]
        return centers

    if row_step_px is not None:
        row_centers = _cluster_fixed_step(ys, row_step_px)
    else:
        row_centers = _cluster_1d(ys, row_gap_scale, min_cluster_size)

    if col_step_px is not None:
        col_centers = _cluster_fixed_step(xs, col_step_px)
    else:
        col_centers = _cluster_1d(xs, col_gap_scale, min_cluster_size)

    row_centers = sorted(row_centers)
    col_centers = sorted(col_centers)

    def _merge_to_expected(centers, expected_count):
        if expected_count is None:
            return centers
        centers = list(sorted(centers))
        while len(centers) > expected_count:
            gaps = [abs(b - a) for a, b in zip(centers[:-1], centers[1:])]
            if not gaps:
                break
            i = int(np.argmin(gaps))
            merged = (centers[i] + centers[i + 1]) / 2.0
            centers = centers[:i] + [merged] + centers[i + 2:]
        return centers

    row_centers = _merge_to_expected(row_centers, expected_rows)
    col_centers = _merge_to_expected(col_centers, expected_cols)

    def _nearest_index(value, centers):
        return int(np.argmin([abs(value - c) for c in centers]))

    holds_out = []
    for idx, h in enumerate(holds):
        row = _nearest_index(h["y"], row_centers)
        col = _nearest_index(h["x"], col_centers)
        holds_out.append(
            {
                "id": idx,
                "x": h["x"],
                "y": h["y"],
                "row": row,
                "col": col,
                "area_shape": h.get("area_shape"),
            }
        )

    payload = {
        "rows": len(row_centers),
        "cols": len(col_centers),
        "row_centers": row_centers,
        "col_centers": col_centers,
        "holds": holds_out,
    }
    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)

    return payload


def load_hold_map(hold_map_path):
    with open(hold_map_path, "r") as f:
        return json.load(f)


def detect_rings(
    image_path,
    hsv_ranges,
    separate_touching=True,
    method="watershed",
    hough_min_radius=6,
    hough_max_radius=24,
    cluster_radius=10,
    hough_param1=120,
    hough_param2=14,
    hough_min_dist=20,
    nms_radius=12,
    ring_coverage=0.35,
    peak_threshold=0.35,
    peak_min_dist=30,
):
    """
    hsv_ranges: dict(name -> (lowerHSV, upperHSV))
    returns: dict(name -> list of (x,y))
    """
    _require_cv2()
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    result = {}
    def _cluster_centers(centers, radius):
        if not centers:
            return centers
        used = [False] * len(centers)
        clustered = []
        for i, (x, y) in enumerate(centers):
            if used[i]:
                continue
            group = [(x, y)]
            used[i] = True
            for j, (x2, y2) in enumerate(centers):
                if used[j]:
                    continue
                if (x - x2) ** 2 + (y - y2) ** 2 <= radius * radius:
                    used[j] = True
                    group.append((x2, y2))
            gx = int(np.mean([p[0] for p in group]))
            gy = int(np.mean([p[1] for p in group]))
            clustered.append((gx, gy))
        return clustered

    for name, hsv_cfg in hsv_ranges.items():
        if isinstance(hsv_cfg, dict):
            lower = hsv_cfg["lower"]
            upper = hsv_cfg["upper"]
            _method = hsv_cfg.get("method", method)
            _separate_touching = hsv_cfg.get("separate_touching", separate_touching)
            _hough_param1 = hsv_cfg.get("hough_param1", hough_param1)
            _hough_param2 = hsv_cfg.get("hough_param2", hough_param2)
            _hough_min_dist = hsv_cfg.get("hough_min_dist", hough_min_dist)
            _hough_min_radius = hsv_cfg.get("hough_min_radius", hough_min_radius)
            _hough_max_radius = hsv_cfg.get("hough_max_radius", hough_max_radius)
            _nms_radius = hsv_cfg.get("nms_radius", nms_radius)
            _ring_coverage = hsv_cfg.get("ring_coverage", ring_coverage)
            _cluster_radius = hsv_cfg.get("cluster_radius", cluster_radius)
        else:
            lower, upper = hsv_cfg
            _method = method
            _separate_touching = separate_touching
            _hough_param1 = hough_param1
            _hough_param2 = hough_param2
            _hough_min_dist = hough_min_dist
            _hough_min_radius = hough_min_radius
            _hough_max_radius = hough_max_radius
            _nms_radius = nms_radius
            _ring_coverage = ring_coverage
            _cluster_radius = cluster_radius
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.medianBlur(mask, 5)

        if _separate_touching and _method == "peaks":
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            if dist.max() <= 0:
                result[name] = []
                continue
            thresh = peak_threshold * dist.max()
            _, peaks = cv2.threshold(dist, thresh, 255, cv2.THRESH_BINARY)
            peaks = peaks.astype(np.uint8)
            # suppress peaks closer than peak_min_dist
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (peak_min_dist, peak_min_dist))
            dilated = cv2.dilate(dist, kernel)
            local_max = (dist == dilated) & (dist >= thresh)
            ys, xs = np.where(local_max)
            centers = list(zip(xs.tolist(), ys.tolist()))
            centers = _cluster_centers(centers, _cluster_radius)
            result[name] = centers
            continue

        if _separate_touching and _method == "watershed":
            # Separate touching rings using distance transform + watershed
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            _, peaks = cv2.threshold(dist, 0.35 * dist.max(), 255, cv2.THRESH_BINARY)
            peaks = peaks.astype(np.uint8)
            num_labels, markers = cv2.connectedComponents(peaks)
            markers = markers + 1
            markers[mask == 0] = 0
            img_copy = img.copy()
            cv2.watershed(img_copy, markers)
            centers = []
            for label in range(2, num_labels + 1):
                ys, xs = np.where(markers == label)
                if len(xs) == 0:
                    continue
                cx = int(np.mean(xs))
                cy = int(np.mean(ys))
                centers.append((cx, cy))
            centers = _cluster_centers(centers, _cluster_radius)
            result[name] = centers
            continue

        if _separate_touching and _method == "hough":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask_closed = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
            blurred = cv2.GaussianBlur(mask_closed, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            circles = cv2.HoughCircles(
                edges,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=_hough_min_dist,
                param1=_hough_param1,
                param2=_hough_param2,
                minRadius=_hough_min_radius,
                maxRadius=_hough_max_radius,
            )
            centers = []
            if circles is not None:
                candidates = [(float(c[0]), float(c[1]), float(c[2])) for c in circles[0]]

                def _ring_overlap_score(cx, cy, r, ring_mask):
                    # Sample points along circle circumference; count how many hit the mask.
                    hits = 0
                    total = 0
                    for angle in np.linspace(0, 2 * np.pi, 36, endpoint=False):
                        x = int(cx + r * np.cos(angle))
                        y = int(cy + r * np.sin(angle))
                        if x < 0 or y < 0 or x >= ring_mask.shape[1] or y >= ring_mask.shape[0]:
                            continue
                        total += 1
                        if ring_mask[y, x] > 0:
                            hits += 1
                    return hits / total if total > 0 else 0.0

                picked = []
                for x, y, r in sorted(candidates, key=lambda t: t[2], reverse=True):
                    if _ring_overlap_score(x, y, r, mask_closed) < _ring_coverage:
                        continue
                    if all((x - px) ** 2 + (y - py) ** 2 > _nms_radius ** 2 for px, py in picked):
                        picked.append((x, y))
                centers = [(int(x), int(y)) for x, y in picked]
            # Fallback: if a ring mask blob has no detected center, add its centroid.
            if True:
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_closed, connectivity=8)
                existing = centers[:]
                for label in range(1, num_labels):
                    x, y, w, h, area = stats[label]
                    if area < 100:
                        continue
                    cx, cy = centroids[label]
                    has_center = False
                    for ex, ey in existing:
                        if x <= ex <= x + w and y <= ey <= y + h:
                            has_center = True
                            break
                    if not has_center:
                        centers.append((int(cx), int(cy)))
            result[name] = centers
            continue

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))
        centers = _cluster_centers(centers, _cluster_radius)
        result[name] = centers
    return result


def map_rings_to_holds(ring_centers, hold_map):
    holds = hold_map["holds"]
    coords = np.array([(h["x"], h["y"]) for h in holds], dtype=np.float32)
    mapped = {}
    for name, centers in ring_centers.items():
        ids = []
        for (x, y) in centers:
            dists = np.sum((coords - np.array([x, y])) ** 2, axis=1)
            idx = int(np.argmin(dists))
            ids.append(holds[idx]["id"])
        mapped[name] = ids
    return mapped


def build_board_matrix(hold_map, ring_hold_ids, channels):
    rows = hold_map["rows"]
    cols = hold_map["cols"]
    mat = np.zeros((rows, cols, len(channels)), dtype=np.float32)
    hold_by_id = {h["id"]: h for h in hold_map["holds"]}
    for ch_i, ch in enumerate(channels):
        for hold_id in ring_hold_ids.get(ch, []):
            h = hold_by_id.get(hold_id)
            if h is None:
                continue
            mat[h["row"], h["col"], ch_i] = 1.0
    return mat


def save_debug_overlay(image_path, hold_map, ring_centers, output_path):
    """
    Draws hold centers, row/col grid lines, and ring centers.
    """
    _require_cv2()
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Draw row/col center lines
    for y in hold_map.get("row_centers", []):
        cv2.line(img, (0, int(y)), (img.shape[1], int(y)), (220, 220, 220), 1)
    for x in hold_map.get("col_centers", []):
        cv2.line(img, (int(x), 0), (int(x), img.shape[0]), (220, 220, 220), 1)

    # Draw all hold centers
    for h in hold_map.get("holds", []):
        cv2.circle(img, (int(h["x"]), int(h["y"])), 2, (120, 120, 120), -1)

    # Ring centers by type
    ring_colors = {
        "start": (0, 255, 0),     # green
        "finish": (255, 0, 255),  # magenta
        "hand": (255, 255, 0),    # cyan/yellow-ish in BGR
        "foot": (0, 140, 255),    # orange
    }
    for name, centers in ring_centers.items():
        color = ring_colors.get(name, (0, 0, 255))
        for (x, y) in centers:
            cv2.circle(img, (int(x), int(y)), 8, color, 2)

    cv2.imwrite(output_path, img)


def extract_metadata_from_label_image(label_image_path):
    """
    Extracts basic metadata from the label crop using OCR.
    Returns dict with raw_text, lines, name, setter, grade_raw, grade_v, stars.
    """
    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        return {
            "raw_text": None,
            "lines": [],
            "name": None,
            "setter": None,
            "grade_raw": None,
            "grade_v": None,
            "stars": None,
        }

    img = Image.open(label_image_path)
    raw_text = pytesseract.image_to_string(img)
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]

    name = lines[0] if len(lines) > 0 else None
    setter = lines[1] if len(lines) > 1 else None

    grade_raw = None
    grade_v = None
    stars = None
    for line in lines:
        if "V" in line:
            grade_raw = line
            # Extract V grade digits
            parts = line.split("V")
            if len(parts) > 1:
                digits = ""
                for ch in parts[1]:
                    if ch.isdigit():
                        digits += ch
                    else:
                        break
                if digits:
                    grade_v = int(digits)
        if "*" in line or "★" in line:
            stars = max(line.count("*"), line.count("★"))

    # Fallback: OCR sometimes reads stars as letters like 'e' or 'o'
    if stars is None and grade_raw is not None:
        tail = grade_raw.lower()
        if "v" in tail:
            tail = tail.split("v", 1)[1]
        # take last token and count letters as star proxy (e.g., "yeo")
        last = tail.strip().split()[-1] if tail.strip() else ""
        last_alpha = "".join([ch for ch in last if ch.isalpha()])
        if 1 <= len(last_alpha) <= 4:
            stars = len(last_alpha)
        else:
            star_like = 0
            for ch in tail:
                if ch in ("e", "o", "y"):
                    star_like += 1
            if star_like > 0:
                stars = min(star_like, 4)

    return {
        "raw_text": raw_text,
        "lines": lines,
        "name": name,
        "setter": setter,
        "grade_raw": grade_raw,
        "grade_v": grade_v,
        "stars": stars,
    }


def export_dataset_json_npy(
    image_dir,
    hold_map_path,
    output_dir,
    hsv_ranges,
    channels,
    method="hough",
):
    """
    Exports per-route matrix (.npy) and metadata (.json).
    Expects board images and label images in the same directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    hold_map = load_hold_map(hold_map_path)

    for filename in sorted(os.listdir(image_dir)):
        if not filename.endswith(".png"):
            continue
        if "label" in filename or "debug_overlay" in filename:
            continue
        board_path = os.path.join(image_dir, filename)
        label_path = os.path.join(image_dir, filename[:-4] + "label_.png")

        ring_centers = detect_rings(
            board_path,
            hsv_ranges,
            separate_touching=True,
            method=method,
            hough_min_radius=26,
            hough_max_radius=34,
            hough_min_dist=40,
            hough_param1=120,
            hough_param2=16,
            nms_radius=20,
            ring_coverage=0.45,
        )
        ring_ids = map_rings_to_holds(ring_centers, hold_map)

        dynamic_channels = [c for c in channels if c in ("start", "finish", "hand", "foot")]
        mat_route = build_board_matrix(hold_map, ring_ids, dynamic_channels)

        rows = hold_map["rows"]
        cols = hold_map["cols"]
        hold_presence = np.zeros((rows, cols), dtype=np.float32)
        hold_size = np.zeros((rows, cols), dtype=np.float32)
        for h in hold_map["holds"]:
            r = h["row"]
            c = h["col"]
            hold_presence[r, c] = 1.0
            area = h.get("area_shape") or 0.0
            hold_size[r, c] = max(hold_size[r, c], float(area))
        if hold_size.max() > 0:
            hold_size = hold_size / hold_size.max()

        extras = []
        if "hold_presence" in channels:
            extras.append(hold_presence[:, :, None])
        if "hold_size" in channels:
            extras.append(hold_size[:, :, None])
        matrix = mat_route if not extras else np.concatenate([mat_route] + extras, axis=2)

        meta = {
            "filename": filename,
            "rows": hold_map["rows"],
            "cols": hold_map["cols"],
            "channels": channels,
            "ring_counts": {k: len(v) for k, v in ring_ids.items()},
        }
        if os.path.exists(label_path):
            meta.update(extract_metadata_from_label_image(label_path))

        base = os.path.splitext(filename)[0]
        np.save(os.path.join(output_dir, base + ".npy"), matrix)
        with open(os.path.join(output_dir, base + ".json"), "w") as f:
            json.dump(meta, f, indent=2)


def extract_orientations_from_image(
    orientation_image_path,
    hold_map_path,
    output_json_path=None,
    max_hold_dist=25,
    min_line_length=8,
    line_gap=8,
    per_hold=True,
    hold_patch_radius=35,
    tail_max_dist=10,
    center_line_dist=6,
    angle_merge_thresh=0.4,
):
    """
    Detects arrow vectors from an annotated orientation image.
    Returns updated hold_map with per-hold orientations (radians).
    Assumes arrow tail is near hold center; direction is from tail to head.
    """
    _require_cv2()
    hold_map = load_hold_map(hold_map_path)
    img = cv2.imread(orientation_image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {orientation_image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    mask = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 8
    )
    mask = cv2.medianBlur(mask, 3)
    edges = cv2.Canny(mask, 40, 120)

    hold_centers = np.array([(h["x"], h["y"]) for h in hold_map["holds"]], dtype=np.float32)
    orientations = {h["id"]: [] for h in hold_map["holds"]}

    def _point_segment_distance(p, a, b):
        ap = p - a
        ab = b - a
        ab_len2 = np.dot(ab, ab)
        if ab_len2 == 0:
            return np.linalg.norm(ap)
        t = np.clip(np.dot(ap, ab) / ab_len2, 0.0, 1.0)
        proj = a + t * ab
        return np.linalg.norm(p - proj)

    def _merge_angles(angles, weights, thresh):
        merged = []
        for ang, w in sorted(zip(angles, weights), key=lambda t: -t[1]):
            if all(abs(np.arctan2(np.sin(ang - m[0]), np.cos(ang - m[0]))) > thresh for m in merged):
                merged.append([ang, w])
        return [m[0] for m in merged]

    if per_hold:
        for h in hold_map["holds"]:
            cx, cy = int(h["x"]), int(h["y"])
            x0 = max(0, cx - hold_patch_radius)
            y0 = max(0, cy - hold_patch_radius)
            x1 = min(edges.shape[1], cx + hold_patch_radius)
            y1 = min(edges.shape[0], cy + hold_patch_radius)
            patch = edges[y0:y1, x0:x1]

            lines = cv2.HoughLinesP(
                patch,
                rho=1,
                theta=np.pi / 180,
                threshold=15,
                minLineLength=min_line_length,
                maxLineGap=line_gap,
            )
            if lines is None:
                continue

            center = np.array([cx, cy], dtype=np.float32)
            angles = []
            weights = []
            for (x1p, y1p, x2p, y2p) in lines[:, 0]:
                p1 = np.array([x1p + x0, y1p + y0], dtype=np.float32)
                p2 = np.array([x2p + x0, y2p + y0], dtype=np.float32)

                # line should pass near the hold center
                if _point_segment_distance(center, p1, p2) > center_line_dist:
                    continue

                d1 = np.linalg.norm(p1 - center)
                d2 = np.linalg.norm(p2 - center)
                tail = p1 if d1 < d2 else p2
                head = p2 if d1 < d2 else p1
                if np.linalg.norm(tail - center) > tail_max_dist:
                    continue
                vec = head - tail
                length = np.linalg.norm(vec)
                if length < 1e-3:
                    continue
                ang = float(np.arctan2(vec[1], vec[0]))
                angles.append(ang)
                weights.append(length)

            if angles:
                merged = _merge_angles(angles, weights, angle_merge_thresh)
                orientations[h["id"]] = merged[:2]
    else:
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=20,
            minLineLength=min_line_length,
            maxLineGap=line_gap,
        )
        candidates = {h["id"]: [] for h in hold_map["holds"]}
        if lines is not None:
            for (x1, y1, x2, y2) in lines[:, 0]:
                p1 = np.array([x1, y1], dtype=np.float32)
                p2 = np.array([x2, y2], dtype=np.float32)
                mid = (p1 + p2) / 2.0

                dists = np.sum((hold_centers - mid) ** 2, axis=1)
                idx = int(np.argmin(dists))
                if dists[idx] > max_hold_dist ** 2:
                    continue
                hold_center = hold_centers[idx]

                # line should pass near the hold center
                if _point_segment_distance(hold_center, p1, p2) > center_line_dist:
                    continue

                d1 = np.sum((p1 - hold_center) ** 2)
                d2 = np.sum((p2 - hold_center) ** 2)
                tail = p1 if d1 < d2 else p2
                head = p2 if d1 < d2 else p1
                if np.linalg.norm(tail - hold_center) > tail_max_dist:
                    continue

                vec = head - tail
                length = np.linalg.norm(vec)
                if length < 1e-3:
                    continue
                angle = float(np.arctan2(vec[1], vec[0]))
                candidates[hold_map["holds"][idx]["id"]].append((angle, length))

        for hid, items in candidates.items():
            if not items:
                continue
            angles = [a for a, _ in items]
            weights = [w for _, w in items]
            merged = _merge_angles(angles, weights, angle_merge_thresh)
            orientations[hid] = merged[:2]

    # attach to hold_map, keep up to 2 orientations per hold
    for h in hold_map["holds"]:
        angs = orientations.get(h["id"], [])
        # de-duplicate near-identical angles
        unique = []
        for a in angs:
            if all(abs(np.arctan2(np.sin(a - u), np.cos(a - u))) > 0.25 for u in unique):
                unique.append(a)
        h["orientations"] = unique[:2]

    if output_json_path:
        with open(output_json_path, "w") as f:
            json.dump(hold_map, f, indent=2)

    return hold_map


def save_orientation_overlay(orientation_image_path, hold_map, output_path):
    """
    Draws detected orientations on top of the orientation image.
    """
    _require_cv2()
    img = cv2.imread(orientation_image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {orientation_image_path}")

    for h in hold_map["holds"]:
        x, y = int(h["x"]), int(h["y"])
        for ang in h.get("orientations", []):
            dx = int(18 * np.cos(ang))
            dy = int(18 * np.sin(ang))
            cv2.arrowedLine(img, (x, y), (x + dx, y + dy), (0, 0, 255), 1, tipLength=0.3)
    cv2.imwrite(output_path, img)


def extract_orientations_from_svg(
    svg_path,
    hold_map_path,
    output_json_path=None,
    arrow_href="_Image2",
    max_hold_dist=25,
    angle_merge_thresh=0.4,
    arrow_cx=28.0,
    arrow_cy=28.0,
    use_symbol_tip=True,
):
    """
    Extracts orientations from an SVG containing arrow symbols.
    Uses the transform matrix of each arrow to compute direction + center.
    """
    import re
    from pathlib import Path
    import xml.etree.ElementTree as ET

    hold_map = load_hold_map(hold_map_path)
    hold_centers = np.array([(h["x"], h["y"]) for h in hold_map["holds"]], dtype=np.float32)
    orientations = {h["id"]: [] for h in hold_map["holds"]}

    tree = ET.parse(svg_path)
    root = tree.getroot()

    # SVG namespace handling
    ns = {"svg": "http://www.w3.org/2000/svg", "xlink": "http://www.w3.org/1999/xlink"}

    # Helper: parse "matrix(a,b,c,d,e,f)"
    def _parse_matrix(s):
        m = re.search(r"matrix\(([^)]+)\)", s)
        if not m:
            return None
        vals = [float(v) for v in m.group(1).replace(",", " ").split()]
        if len(vals) != 6:
            return None
        return vals  # a,b,c,d,e,f

    # Helper: parse "56px" -> 56
    def _parse_len(s, default):
        if s is None:
            return default
        try:
            return float(re.sub(r"[^0-9.]+", "", s))
        except Exception:
            return default

    # Auto-detect arrow symbol if not provided
    if arrow_href is None:
        href_counts = {}
        for g in root.iter():
            if not g.tag.endswith("g") or not g.get("transform"):
                continue
            for child in g:
                if not child.tag.endswith("use"):
                    continue
                href = child.get("{http://www.w3.org/1999/xlink}href") or child.get("href")
                if href:
                    href_counts[href] = href_counts.get(href, 0) + 1
        if href_counts:
            # pick most common href
            arrow_href = max(href_counts.items(), key=lambda kv: kv[1])[0].lstrip("#")

    # Try to extract the arrow symbol image and compute a local tip/tail
    tip_local = None
    tail_local = None
    if use_symbol_tip and arrow_href:
        svg_text = Path(svg_path).read_text()
        pattern = rf'id="{re.escape(arrow_href)}"[^>]*(xlink:href|href)="data:image/png;base64,([^"]+)"'
        m = re.search(pattern, svg_text)
        if m:
            import base64, io
            from PIL import Image
            b64 = m.group(2)
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")
            A = np.array(img.split()[-1])
            ys, xs = np.where(A > 0)
            if len(xs) > 10:
                pts = np.stack([xs, ys], axis=1).astype(np.float32)
                # PCA for principal axis
                mean = pts.mean(axis=0)
                X = pts - mean
                cov = X.T @ X / max(len(X) - 1, 1)
                eigvals, eigvecs = np.linalg.eigh(cov)
                v = eigvecs[:, np.argmax(eigvals)]
                v = v / (np.linalg.norm(v) + 1e-9)
                proj = X @ v
                pmin, pmax = proj.min(), proj.max()
                # take end slices
                span = pmax - pmin
                end_thresh = 0.15 * span
                near_min = pts[proj <= (pmin + end_thresh)]
                near_max = pts[proj >= (pmax - end_thresh)]

                # width (perpendicular spread) at each end
                def end_width(end_pts):
                    if len(end_pts) == 0:
                        return 0.0
                    Xp = end_pts - mean
                    perp = np.array([-v[1], v[0]])
                    return float(np.std(Xp @ perp))

                w_min = end_width(near_min)
                w_max = end_width(near_max)
                # arrowhead typically wider -> tip
                if w_max >= w_min:
                    tip_local = near_max.mean(axis=0)
                    tail_local = near_min.mean(axis=0)
                else:
                    tip_local = near_min.mean(axis=0)
                    tail_local = near_max.mean(axis=0)

    # Fallback to centroid if tip/tail not found
    if tip_local is None or tail_local is None:
        tip_local = np.array([arrow_cx + 8.0, arrow_cy], dtype=np.float32)
        tail_local = np.array([arrow_cx - 8.0, arrow_cy], dtype=np.float32)

    # Find <g> elements with transform and child <use xlink:href="#_ImageX">
    for g in root.iter():
        if not g.tag.endswith("g"):
            continue
        transform = g.get("transform")
        if not transform:
            continue
        matrix = _parse_matrix(transform)
        if matrix is None:
            continue
        a, b, c, d, e, f = matrix

        use = None
        for child in g:
            if not child.tag.endswith("use"):
                continue
            href = child.get("{http://www.w3.org/1999/xlink}href")
            if href and href.endswith(arrow_href):
                use = child
                break
        if use is None:
            continue

        w = _parse_len(use.get("width"), 56.0)
        h = _parse_len(use.get("height"), 56.0)
        # Use calibrated arrow centroid (not geometric center)
        cx_local = arrow_cx * (w / 56.0)
        cy_local = arrow_cy * (h / 56.0)

        # Transform local tip/tail to global
        tip = np.array([tip_local[0] * (w / 56.0), tip_local[1] * (h / 56.0)], dtype=np.float32)
        tail = np.array([tail_local[0] * (w / 56.0), tail_local[1] * (h / 56.0)], dtype=np.float32)
        tip_g = np.array([a * tip[0] + c * tip[1] + e, b * tip[0] + d * tip[1] + f], dtype=np.float32)
        tail_g = np.array([a * tail[0] + c * tail[1] + e, b * tail[0] + d * tail[1] + f], dtype=np.float32)

        # Orientation angle
        angle = float(np.arctan2(tip_g[1] - tail_g[1], tip_g[0] - tail_g[0]))

        # map to nearest hold
        dists = np.sum((hold_centers - tail_g) ** 2, axis=1)
        idx = int(np.argmin(dists))
        if dists[idx] > max_hold_dist ** 2:
            continue
        orientations[hold_map["holds"][idx]["id"]].append(angle)

    # merge near-duplicates; keep up to 2
    for h in hold_map["holds"]:
        angs = orientations.get(h["id"], [])
        unique = []
        for a in angs:
            if all(abs(np.arctan2(np.sin(a - u), np.cos(a - u))) > angle_merge_thresh for u in unique):
                unique.append(a)
        h["orientations"] = unique[:2]

    if output_json_path:
        with open(output_json_path, "w") as f:
            json.dump(hold_map, f, indent=2)

    return hold_map

# Press the green button in the gutter to run the script.

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Kilterboard dataset utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_map = sub.add_parser("build-hold-map", help="Build hold map from empty board image")
    p_map.add_argument("--empty-board", required=True)
    p_map.add_argument("--output-json", required=True)
    p_map.add_argument("--row-step", type=int, default=25)
    p_map.add_argument("--col-step", type=int, default=25)

    p_debug = sub.add_parser("debug-overlay", help="Generate debug overlay image")
    p_debug.add_argument("--board-image", required=True)
    p_debug.add_argument("--hold-map", required=True)
    p_debug.add_argument("--output", required=True)

    p_export = sub.add_parser("export", help="Export matrices + metadata as NPY/JSON")
    p_export.add_argument("--image-dir", required=True)
    p_export.add_argument("--hold-map", required=True)
    p_export.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    if args.cmd == "build-hold-map":
        build_hold_map(
            args.empty_board,
            args.output_json,
            use_bolt_holes=True,
            row_step_px=args.row_step,
            col_step_px=args.col_step,
        )
        print(f"Wrote hold map to {args.output_json}")

    elif args.cmd == "debug-overlay":
        hold_map = load_hold_map(args.hold_map)
        hsv_ranges = {
            "start": {
                "lower": [35, 80, 80],
                "upper": [85, 255, 255],
                "hough_param2": 12,
                "hough_min_dist": 32,
                "ring_coverage": 0.35,
                "nms_radius": 16,
            },
            "finish": ([145, 80, 80], [175, 255, 255]),
            "hand": ([80, 80, 80], [100, 255, 255]),
            "foot": ([10, 80, 80], [25, 255, 255]),
        }
        ring_centers = detect_rings(
            args.board_image,
            hsv_ranges,
            separate_touching=True,
            method="hough",
            hough_min_radius=26,
            hough_max_radius=34,
            hough_min_dist=40,
            hough_param1=120,
            hough_param2=16,
            nms_radius=20,
            ring_coverage=0.45,
        )
        save_debug_overlay(args.board_image, hold_map, ring_centers, args.output)
        print(f"Wrote debug overlay to {args.output}")

    elif args.cmd == "export":
        hsv_ranges = {
            "start": {
                "lower": [35, 80, 80],
                "upper": [85, 255, 255],
                "hough_param2": 12,
                "hough_min_dist": 32,
                "ring_coverage": 0.35,
                "nms_radius": 16,
            },
            "finish": ([145, 80, 80], [175, 255, 255]),
            "hand": {
                "lower": [80, 80, 80],
                "upper": [100, 255, 255],
                "method": "hough",
                "hough_min_dist": 26,
                "hough_param2": 10,
                "ring_coverage": 0.30,
                "nms_radius": 12,
            },
            "foot": ([10, 80, 80], [25, 255, 255]),
        }
        export_dataset_json_npy(
            args.image_dir,
            args.hold_map,
            args.output_dir,
            hsv_ranges,
            channels=["start", "finish", "hand", "foot", "hold_presence", "hold_size"],
            method="hough",
        )
        print(f"Exported dataset to {args.output_dir}")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
