import io
import os
import uuid
import zipfile
from collections import OrderedDict

import PyPDF2
import cv2
import jsonpickle
import numpy as np
from cv2.cv2 import KeyPoint

from core.models import Image

FILE_PATH = 'core/static/imgs/'
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", '.swg', '.gif', '.tiff')
MAX_FEATURES = 500
INPUT_FOLDER = 'core/static/imgs/input_images'
COLLECTION_FOLDER = 'core/static/imgs/imgs_collection'
THRESHOLD = 70


def recurse(xObject, img_dir, counter):
    xObject = xObject['/Resources']['/XObject'].getObject()
    unique_filenames = []
    for obj in xObject:

        if xObject[obj]['/Subtype'] == '/Image':
            size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
            data = xObject[obj]._data
            if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
                mode = "RGB"
            else:
                mode = "P"
            unique_filename = str(uuid.uuid4())
            imagename = FILE_PATH + os.path.join(
                img_dir, os.path.basename(unique_filename))

            if xObject[obj]['/Filter'] == '/FlateDecode':
                imagename += '.png'
                unique_filename += '.png'
                img = Image.frombytes(mode, size, data)
                img.save(imagename)
                counter += 1
            elif xObject[obj]['/Filter'] == '/DCTDecode':
                imagename += '.jpg'
                unique_filename += '.jpg'
                img = open(imagename, "wb")
                img.write(data)
                img.close()
                counter += 1
            elif xObject[obj]['/Filter'] == '/JPXDecode':
                imagename += '.jp2'
                unique_filename += '.jp2'
                img = open(imagename, "wb")
                img.write(data)
                img.close()
                counter += 1
            unique_filenames.append(unique_filename)
    return counter, unique_filenames


def extract_from_pdf(incoming_file, img_dir, store_to_db):
    pdf = PyPDF2.PdfFileReader(io.BytesIO(incoming_file.read()))
    pages = pdf.getNumPages()
    counter = 0
    imgs = []
    for p in range(pages):
        page = pdf.getPage(p)
        counter, imagename = recurse(page, img_dir, counter)
        if imagename:
            imgs.extend(imagename)
    if store_to_db:
        store_imgs_to_db(imgs, img_dir)
    return counter


def extract_images(incoming_file, img_dir, store_to_db=False):
    if incoming_file.content_type == 'application/pdf':
        return extract_from_pdf(incoming_file, img_dir, store_to_db)
    zipf = zipfile.ZipFile(incoming_file)
    filelist = zipf.namelist()
    image_counter = 0
    imgs = []
    for fname in filelist:
        _, extension = os.path.splitext(fname)
        if extension in IMAGE_EXTENSIONS:
            unique_filename = str(uuid.uuid4()) + extension
            if store_to_db:
                dst_fname = os.path.join(img_dir, os.path.basename(unique_filename))
            else:
                dst_fname = os.path.join(img_dir, os.path.basename(fname))
            imgs.append(unique_filename)
            dst_fname = FILE_PATH + dst_fname
            with open(dst_fname, "wb") as dst_f:
                dst_f.write(zipf.read(fname))
            image_counter += 1
    zipf.close()
    if store_to_db:
        store_imgs_to_db(imgs, img_dir)
    return image_counter


def store_imgs_to_db(images, img_dir):
    for img_name in images:
        img = cv2.imread(FILE_PATH + img_dir + '/' + img_name, 0)
        detector = get_detector('surf')
        keypoints, descriptors = detector.detectAndCompute(img, None)
        keypoints_json = []
        for keypoint in keypoints:  # type: KeyPoint
            keypoints_json.append({
                'x': keypoint.pt[0],
                'y': keypoint.pt[1],
                '_size': keypoint.size,
                '_angle': keypoint.angle,
                '_response': keypoint.response,
                '_octave': keypoint.octave,
                '_class_id': keypoint.class_id,
            })
        keypoints_json = jsonpickle.encode(keypoints_json)
        descriptors_json = jsonpickle.encode(descriptors)
        Image.objects.create(title=img_name, keypoints=keypoints_json,
                             descriptors=descriptors_json)


def remove_files(folder: str):
    files = os.listdir(folder)
    for file in files:
        os.remove(folder + '/' + file)


def research(keypoints: list, matches: list, inliers: list, similarity: list):
    def _fit(values):
        result_dict = {}
        for res in values:
            for key, value in res.items():
                new_key = key.replace('.jpg', '').replace('.jpeg', '').replace('.png', '').replace('.bmp', '')
                result_dict.update({new_key: value})
        ordered_result = OrderedDict(
            sorted(result_dict.items(), key=lambda t: t[0]))
        return ordered_result

    ordered_keys = _fit(keypoints)
    ordered_matches = _fit(matches)
    ordered_inliers = _fit(inliers)
    ordered_similarity = _fit(similarity)

    print('keys:', ordered_keys)
    print('matches:', ordered_matches)
    print('inliers:', ordered_inliers)
    print('similarity:', ordered_similarity)


def run_image_comparison():
    input_files = os.listdir(INPUT_FOLDER)
    collection_images = Image.objects.all()
    CALCS = {}
    res_print = []
    res_print_key = []
    res_print_pairs = []
    res_print_inliers = []
    for img1_p in input_files:
        for img_collection in collection_images:
            img1_path = f'{INPUT_FOLDER}/{img1_p}'
            print(f'~~~~~~~{img1_p}~~~~~~~')
            img1 = cv2.imread(img1_path, 0)
            keypoints_pairs, keypoints1, keypoints2 = match_images_collection(
                img1, img_collection)
            min_keypoints = min(len(keypoints1), len(keypoints2))
            perc = len(keypoints_pairs) / min_keypoints
            print(f'Keypoints border:{perc * 100}%')
            if keypoints_pairs and perc > 0.25:
                H, mask = build_homography(keypoints_pairs)
                if mask is not None:
                    inliers = np.sum(mask)
                    matched = len(mask)
                    calculations = round((inliers / matched) * 100, 2)
                    print(calculations)
                    res_print.append({img1_p: calculations})
                    res_print_key.append({img1_p: len(keypoints1)})
                    res_print_pairs.append({img1_p: len(keypoints_pairs)})
                    res_print_inliers.append({img1_p: inliers})
                    CALCS.setdefault(f'imgs/input_images/{img1_p}', []).append(
                        [f'imgs/imgs_collection/{img_collection.title}',
                         calculations])
    results = {}
    for key, value in CALCS.items():
        value.sort(key=lambda x: x[1])
        for url, val in value:
            if val > THRESHOLD:
                results[key] = (url, val)
    research(res_print_key, res_print_pairs, res_print_inliers, res_print)
    return results


def get_detector(name):
    if name == 'sift':
        return cv2.xfeatures2d.SIFT_create()
    if name == 'surf':
        return cv2.xfeatures2d.SURF_create(1000, 5, 5, extended=True)
        # return cv2.xfeatures2d.SURF_create(400, extended=False)
    if name == 'orb':
        return cv2.ORB_create(MAX_FEATURES)


def match_images_collection(img1, img_collection: Image):
    """Takes two image, where the second one is an collection img object"""
    detector = get_detector('surf')
    matcher = cv2.BFMatcher(normType=cv2.NORM_L2)

    keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
    keypoints2_raw = jsonpickle.decode(img_collection.keypoints)
    keypoints2 = []
    for keypoint in keypoints2_raw:
        keypoints2.append(cv2.KeyPoint(**keypoint))
    descriptors2 = jsonpickle.decode(img_collection.descriptors)
    print(
        f'img1: {len(keypoints1)} features, img2: {len(keypoints2)} features')
    raw_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    keypoints_pairs = filter_matches(keypoints1, keypoints2, raw_matches)
    return keypoints_pairs, keypoints1, keypoints2


def filter_matches(kp1, kp2, matches, ratio=0.75):
    keypoints_pairs = []
    for match in matches:
        if match[0].distance < match[1].distance * ratio:
            match1 = match[0]
            keypoints_pairs.append(
                (kp1[match1.queryIdx], kp2[match1.trainIdx]))
    print(f'pairs: {len(keypoints_pairs)}')
    return keypoints_pairs


def build_homography(keypoints_pairs):
    match_keypoints1, match_keypoints2 = zip(*keypoints_pairs)

    points1 = np.float32([kp.pt for kp in match_keypoints1])
    points2 = np.float32([kp.pt for kp in match_keypoints2])
    if len(keypoints_pairs) > 20:
        H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
    else:
        H, mask = None, None
        print(
            f'{len(points1)} matches found, not enough for homography estimation')

    return H, mask
