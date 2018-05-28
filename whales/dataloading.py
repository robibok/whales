import hashlib
import io
import random
import traceback
from bunch import Bunch
from math import floor
from skimage import transform
from skimage.transform import warp, SimilarityTransform, AffineTransform, estimate_transform
from TheanoLib import utils
from image_utils import resize_simple, fetch_path_local
import numpy as np
from ml_utils import floatX, unpack, start_timer
import ml_utils

from augmentation import fast_warp, build_center_uncenter_transforms2, random_perturbation_transform
from augmentation import perturb  as perturb_fun
from whales.main import WhaleTrainer

# Different order of pca, std, mean, historical reasons
def transformation_historical(img, spec, perturb):
    # inter_size = spec['inter_size']
    mean = spec['mean']
    std = spec['std']
    img = img.astype(dtype=floatX)
    # img /= 255.0

    def apply_mean(img):
        if mean is not None:
            assert (len(mean) == spec['target_channels'])
            for channel in xrange(spec['target_channels']):
                img[:, :, channel] -= mean[channel]

        return img

    def apply_std(img):
        if std is not None:
            assert (len(std) == spec['target_channels'])
            for channel in xrange(spec['target_channels']):
                img[:, :, channel] /= std[channel]
        return img


    if perturb:
        img = perturb_fun(img, spec['augmentation_params'], target_shape=(spec['target_h'], spec['target_w']))

    # imgs.append(img)
    # img = np.copy(img)

    # PCA
    apply_std(img)
    if spec['pca_data'] is not None:
        evs, U = spec['pca_data']
        ls = evs.astype(float) * np.random.normal(scale=spec['pca_scale'], size=evs.shape[0])
        noise = U.dot(ls).reshape((1, 1, evs.shape[0]))
        # print evs, ls, U
        # print 'noise', noise
        img += noise
    img = apply_mean(img)

    def f(img):
        img = np.rollaxis(img, 2)
        return img

    # The img was H x W x C before
    return f(img)


def transformation(img, spec, perturb):
    # inter_size = spec['inter_size']
    mean = spec['mean']
    std = spec['std']
    img = img.astype(dtype=floatX)
    # img /= 255.0

    def apply_mean_std(img):
        if mean is not None:
            assert (len(mean) == spec['target_channels'])
            for channel in xrange(spec['target_channels']):
                img[:, :, channel] -= mean[channel]

        if std is not None:
            assert (len(std) == spec['target_channels'])
            for channel in xrange(spec['target_channels']):
                img[:, :, channel] /= std[channel]
        return img

    if perturb:
        img = perturb_fun(img, spec['augmentation_params'], target_shape=(spec['target_h'], spec['target_w']))

    # imgs.append(img)
    # img = np.copy(img)

    # PCA
    if spec['pca_data'] is not None:
        evs, U = spec['pca_data']
        ls = evs.astype(float) * np.random.normal(scale=spec['pca_scale'], size=evs.shape[0])
        noise = U.dot(ls).reshape((1, 1, evs.shape[0]))
        # print evs, ls, U
        # print 'noise', noise
        img += noise

    img = apply_mean_std(img)

    def f(img):
        img = np.rollaxis(img, 2)
        return img

    # The img was H x W x C before
    return f(img)

def print_traceback(f):
    def g(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            print traceback.format_exc()
            raise
    return g


def find_bucket(s, buckets, wsp):
            if wsp < 0 or wsp >= s:
                return -1
            res = int(floor((wsp * buckets) / s))
            assert(res >= 0 and res < buckets)
            return res


def rev_find_bucket(s, buckets, wsp):
    return (wsp * s) / buckets

def rev_find_bucket_robert(s, buckets, wsp):
    raise RuntimeError()


def fetch_example_crop2(recipe, global_spec):
    try:
        img = fetch_path_local(recipe.path)
        target_h = global_spec['target_h']
        target_w = global_spec['target_w']
        TARGETS = global_spec['TARGETS']
        buckets = global_spec['buckets']
        equalize = global_spec['equalize']
        target_size = target_h
        if target_h != target_w:
            raise NotImplementedError()

        true_dist = np.zeros(shape=(WhaleTrainer.get_y_shape(TARGETS),), dtype=floatX)



        #img = resize_simple(img, target_h)
        crop_annos = recipe.annotations['auto_slot']
        idx = random.randint(0, len(crop_annos) - 1)
        crop_anno = crop_annos[idx]
        (x1, y1) = crop_anno['coord1']
        (x2, y2) = crop_anno['coord2']
        x1 *= 4
        y1 *= 4
        x2 *= 4
        y2 *= 4


        if 'point1' in recipe.annotations:
            point_1 = recipe.annotations['point1'][0]
        else:
            point_1 = {'x': 0.0, 'y': 0.0}

        if 'point2' in recipe.annotations:
            point_2 = recipe.annotations['point2'][0]
        else:
            point_2 = {'x': 0.0, 'y': 0.0}

        if 'class_idx' in recipe:
            inter = WhaleTrainer.get_interval('class', TARGETS)
            assert(inter is not None)
            assert(inter[0] == 0)
            #print 'inter0', inter
            true_dist[inter[0] + recipe.class_idx] = 1.0

        inter = WhaleTrainer.get_interval('widacryj', TARGETS)
        if 'widacryj' in recipe.annotations and inter is not None:
            widacryj_target = recipe.annotations['widacryj']
            assert (widacryj_target >= 0 and widacryj_target < inter[1] - inter[0])
            true_dist[inter[0] + widacryj_target] = 1

        inter = WhaleTrainer.get_interval('new_conn', TARGETS)
        if 'new_conn' in recipe.annotations and inter is not None:
            new_conn_target_a = recipe.annotations['new_conn']
            if new_conn_target_a == 0:
                new_conn_target = 0
            elif new_conn_target_a == 2:
                new_conn_target = 1
            else:
                new_conn_target = -1

            if new_conn_target != -1:
                #print 'New_conn', new_conn_target
                assert (new_conn_target >= 0 and new_conn_target < inter[1] - inter[0])
                true_dist[inter[0] + new_conn_target] = 1

        inter = WhaleTrainer.get_interval('symetria', TARGETS)
        if 'symetria' in recipe.annotations and inter is not None:
            symetria_target = recipe.annotations['symetria']

            if symetria_target != -1:
                assert (symetria_target >= 0 and symetria_target < inter[1] - inter[0])
                true_dist[inter[0] + symetria_target] = 1

        ok_x, ok_y = x1, y1
        ann_w, ann_h = x2 - x1, y2 - y1

        #print 'before',recipe.name, ok_x, ok_y, ann_w, ann_h
        if equalize:
            s = max(ann_w, ann_h)
            ok_x = (ok_x + ann_w / 2) - s / 2
            ok_y = (ok_y + ann_h / 2) - s / 2
            ann_w = s
            ann_h = s


        # print ok_x, ok_y, ann_w, ann_h
        # print target_w, target_h
        # print img.shape
        if ann_h == 0 or ann_w == 0:
            print 'BAD, ann_h = 0 or ann_w = 0', recipe.name
            ann_h = max(ann_h, 1)
            ann_w = max(ann_w, 1)

        #print 'rest1 took', ml_utils.elapsed_time_ms(timer)
        tform_res = SimilarityTransform(translation=(-ok_x, -ok_y))
        tform_res += AffineTransform(scale=(target_w / float(ann_w), target_h / float(ann_h)))
        # img is in (0, 0), (target_w, target_h)

        tform_center, tform_uncenter = build_center_uncenter_transforms2(target_w, target_h)
        tform_augment, r = unpack(random_perturbation_transform(rng=np.random, **global_spec['augmentation_params']),
                                  'tform', 'r')

        #print 'after',recipe.name, ok_x, ok_y, ann_w, ann_h, r
        tform_res += tform_center + tform_augment + tform_uncenter

        img = fast_warp(img, AffineTransform(tform_res._inv_matrix), output_shape=(target_h, target_w))
        img = transformation(img, global_spec, perturb=False)

        res1 = tform_res((point_1['x'], point_1['y']))[0]
        res2 = tform_res((point_2['x'], point_2['y']))[0]

        bucket1_x = find_bucket(target_w, buckets, res1[0])
        bucket1_y = find_bucket(target_h, buckets, res1[1])
        bucket2_x = find_bucket(target_w, buckets, res2[0])
        bucket2_y = find_bucket(target_h, buckets, res2[1])

        inter = WhaleTrainer.get_interval('indygo_point1_x', TARGETS)
        if bucket1_x != -1 and bucket1_y != -1 and inter is not None:
            idx = inter[0] + bucket1_x
            if idx < inter[1]:
                true_dist[idx] = 1

        inter = WhaleTrainer.get_interval('indygo_point1_y', TARGETS)
        if bucket1_x != -1 and bucket1_y != -1 and inter is not None:
            idx = inter[0] + bucket1_y
            if idx < inter[1]:
                true_dist[idx] = 1

        inter = WhaleTrainer.get_interval('indygo_point2_x', TARGETS)
        if bucket2_x != -1 and bucket2_y != -1 and inter is not None:
            idx = inter[0] + bucket2_x
            if idx < inter[1]:
                true_dist[idx] = 1

        inter = WhaleTrainer.get_interval('indygo_point2_y', TARGETS)
        if bucket2_x != -1 and bucket2_y != -1 and inter is not None:
            idx = inter[0] + bucket2_y
            #print idx
            if idx < inter[1]:
                true_dist[idx] = 1

        info = {'tform_res': tform_res,
                'r': r
                }

        inter = WhaleTrainer.get_interval('conn', TARGETS)
        if 'ryj_conn' in recipe.annotations and inter is not None:
            ryj_conn_anno = recipe.annotations['ryj_conn']
            class_idx = ryj_conn_anno['class']
            true_dist[inter[0] + class_idx] = 1

        def draw_point(img, x, y, color, w=7):
            img[0, y:y+w, x:x+w] = color[0]
            img[1, y:y+w, x:x+w] = color[1]
            img[2, y:y+w, x:x+w] = color[2]

        target_size = target_h
        x1 = (float(target_size) / buckets) * bucket1_x
        y1 = (float(target_size) / buckets) * bucket1_y
        x2 = (float(target_size) / buckets) * bucket2_x
        y2 = (float(target_size) / buckets) * bucket2_y
        w = int(floor(target_size / buckets))
        #draw_point(img, x1, y1, (0, 0, 2), w=w)
        #draw_point(img, x2, y2, (2, 0, 0), w=w)

        #draw_point(img, res1[0], res1[1], (0, 0, 2))
        #draw_point(img, res2[0], res2[1], (2, 0, 0))
        #print true_dist[447+2: 447+2+30]

        #print 'rest4 took', ml_utils.elapsed_time_ms(timer)
        return Bunch(x=img, y=true_dist, recipe=recipe, info=info)
    except Exception as e:
            print traceback.format_exc()
            raise



def fetch_example_anno_indygo(recipe, global_spec):
    try:
        img = fetch_path_local(recipe.path)
        target_h = global_spec['target_h']
        target_w = global_spec['target_w']
        TARGETS = global_spec['TARGETS']
        buckets = global_spec['buckets']

        target_size = target_h
        indygo_equalize = global_spec['indygo_equalize']
        m = global_spec['margin']
        diag = global_spec['diag']
        if target_h != target_w:
            raise NotImplementedError()

        true_dist = np.zeros(shape=(WhaleTrainer.get_y_shape(TARGETS),), dtype=floatX)

        if 'class_idx' in recipe:
            inter = WhaleTrainer.get_interval('class', TARGETS)
            assert(inter is not None)
            assert(inter[0] == 0)
            #print 'inter0', inter
            true_dist[inter[0] + recipe.class_idx] = 1.0


        #img = resize_simple(img, target_h)
        crop_annos = recipe.annotations['auto_indygo']
        idx = random.randint(0, len(crop_annos) - 1)
        crop_anno = crop_annos[idx]
        (x1, y1) = crop_anno['coord1']
        (x2, y2) = crop_anno['coord2']

        if 'point1' in recipe.annotations:
            point_1 = recipe.annotations['point1'][0]
        else:
            point_1 = {'x': 0.0, 'y': 0.0}

        if 'point2' in recipe.annotations:
            point_2 = recipe.annotations['point2'][0]
        else:
            point_2 = {'x': 0.0, 'y': 0.0}

        inter = WhaleTrainer.get_interval('widacryj', TARGETS)

        if 'widacryj' in recipe.annotations and inter is not None:
            widacryj_target = recipe.annotations['widacryj']
            assert (widacryj_target >= 0 and widacryj_target < inter[1] - inter[0])
            true_dist[inter[0] + widacryj_target] = 1

        inter = WhaleTrainer.get_interval('new_conn', TARGETS)
        if 'new_conn' in recipe.annotations and inter is not None:
            new_conn_target_a = recipe.annotations['new_conn']
            if new_conn_target_a == 0:
                new_conn_target = 0
            elif new_conn_target_a == 2:
                new_conn_target = 1
            else:
                new_conn_target = -1

            if new_conn_target != -1:
                #print 'New_conn', new_conn_target
                assert (new_conn_target >= 0 and new_conn_target < inter[1] - inter[0])
                true_dist[inter[0] + new_conn_target] = 1

        inter = WhaleTrainer.get_interval('symetria', TARGETS)
        if 'symetria' in recipe.annotations and inter is not None:
            symetria_target = recipe.annotations['symetria']

            if symetria_target != -1:
                assert (symetria_target >= 0 and symetria_target < inter[1] - inter[0])
                true_dist[inter[0] + symetria_target] = 1

        if diag:
            dsts = [[m, m], [target_size - m, target_size - m]]
        else:
            dsts = [[target_size / 2, m], [target_size / 2, target_size - m]]

        srcs = [[x1, y1], [x2, y2]]

        def rot90(w):
            return np.array([-w[1], w[0]], dtype=w.dtype)

        if indygo_equalize:
            s1 = np.array(srcs[0])
            s2 = np.array(srcs[1])
            w = s2 - s1
            wp = rot90(w)

            d1 = np.array(dsts[0])
            d2 = np.array(dsts[1])
            v = d2 - d1
            vp = rot90(v)
            srcs.append(list(s1 + wp))
            dsts.append(list(d1 + vp))
            # print '-------------'
            # print 'srcs'
            # print srcs
            # print 'dsts'
            # print dsts
            # we want (wxp, wyp) transleted to

        src = np.array(srcs)
        dst = np.array(dsts)

        tform_res = estimate_transform('affine', src, dst)

        tform_center, tform_uncenter = build_center_uncenter_transforms2(target_w, target_h)
        tform_augment, r = unpack(random_perturbation_transform(rng=np.random, **global_spec['augmentation_params']),
                                  'tform', 'r')

        tform_res += tform_center + tform_augment + tform_uncenter
        img = fast_warp(img, AffineTransform(tform_res._inv_matrix), output_shape=(target_h, target_w))
        img = transformation(img, global_spec, perturb=False)

        res1 = tform_res((point_1['x'], point_1['y']))[0]
        res2 = tform_res((point_2['x'], point_2['y']))[0]


        bucket1_x = find_bucket(target_w, buckets, res1[0])
        bucket1_y = find_bucket(target_h, buckets, res1[1])
        bucket2_x = find_bucket(target_w, buckets, res2[0])
        bucket2_y = find_bucket(target_h, buckets, res2[1])

        inter = WhaleTrainer.get_interval('indygo_point1_x', TARGETS)
        if bucket1_x != -1 and bucket1_y != -1 and inter is not None:
            idx = inter[0] + bucket1_x
            if idx < inter[1]:
                true_dist[idx] = 1

        inter = WhaleTrainer.get_interval('indygo_point1_y', TARGETS)
        if bucket1_x != -1 and bucket1_y != -1 and inter is not None:
            idx = inter[0] + bucket1_y
            if idx < inter[1]:
                true_dist[idx] = 1

        inter = WhaleTrainer.get_interval('indygo_point2_x', TARGETS)
        if bucket2_x != -1 and bucket2_y != -1 and inter is not None:
            idx = inter[0] + bucket2_x
            if idx < inter[1]:
                true_dist[idx] = 1

        inter = WhaleTrainer.get_interval('indygo_point2_y', TARGETS)
        if bucket2_x != -1 and bucket2_y != -1 and inter is not None:
            idx = inter[0] + bucket2_y
            #print idx
            if idx < inter[1]:
                true_dist[idx] = 1

        info = {
                'perturb_params': r
                }
        #print 'buckets', res1, bucket1_x, bucket1_y


        inter = WhaleTrainer.get_interval('conn', TARGETS)
        if 'ryj_conn' in recipe.annotations and inter is not None:
            ryj_conn_anno = recipe.annotations['ryj_conn']
            class_idx = ryj_conn_anno['class']
            true_dist[inter[0] + class_idx] = 1

        def draw_point(img, x, y, color, w=7):
            img[0, y:y+w, x:x+w] = color[0]
            img[1, y:y+w, x:x+w] = color[1]
            img[2, y:y+w, x:x+w] = color[2]

        target_size = target_h
        x1 = (float(target_size) / buckets) * bucket1_x
        y1 = (float(target_size) / buckets) * bucket1_y
        x2 = (float(target_size) / buckets) * bucket2_x
        y2 = (float(target_size) / buckets) * bucket2_y
        w = int(floor(target_size / buckets))
        #draw_point(img, x1, y1, (0, 0, 2), w=w)
        #draw_point(img, x2, y2, (2, 0, 0), w=w)

        #draw_point(img, res1[0], res1[1], (0, 0, 2))
        #draw_point(img, res2[0], res2[1], (2, 0, 0))
        #print true_dist[447+2: 447+2+30]

        #print 'rest4 took', ml_utils.elapsed_time_ms(timer)
        return Bunch(x=img, y=true_dist, recipe=recipe, info=info)
    except Exception as e:
            print traceback.format_exc()
            raise


def fetch_rob_crop_recipe(recipe, global_spec):
    try:
        img = fetch_path_local(recipe.path)
        img_h = img.shape[0]
        img_w = img.shape[1]
        pre_h, pre_w = 256, 256
        target_h, target_w = global_spec['target_h'], global_spec['target_w']
        TARGETS = global_spec['TARGETS']
        buckets = global_spec['buckets']

        sx = (pre_w - target_w + 1) / 2
        sy = (pre_h - target_h + 1) / 2
        #print sx, sy

        #target_h, target_w = 256, 256
        #print 'img_size', img.shape
        true_dist = np.zeros(shape=(WhaleTrainer.get_y_shape(TARGETS),), dtype=floatX)

        tform_res = AffineTransform(scale=(pre_w / float(img_w), pre_h / float(img_h)))
        tform_res += SimilarityTransform(translation=(-sx, -sy))

        tform_augment, r = unpack(random_perturbation_transform(rng=np.random, **global_spec['augmentation_params']),
                                  'tform', 'r')

        tform_center, tform_uncenter = build_center_uncenter_transforms2(target_w, target_h)
        tform_res += tform_center + tform_augment + tform_uncenter

        img = fast_warp(img, AffineTransform(tform_res._inv_matrix), output_shape=(target_h, target_w))
        img = transformation(img, global_spec, perturb=False)

        if recipe.fetch_true_dist:
            # Constructing true_dist

            def go_indygo(name, a, b, v):
                inter = WhaleTrainer.get_interval(name, TARGETS)
                if a != -1 and b != -1 and inter is not None:
                    idx = inter[0] + a
                    if idx < inter[1]:
                        true_dist[idx] = v

            # slot
            slot_resize_scale = 0.25
            slot_annotation = recipe.annotations['slot'][0]
            ann1_x, ann1_y, ann_w, ann_h = (slot_annotation['x'] * slot_resize_scale,
                                          slot_annotation['y'] * slot_resize_scale,
                                          slot_annotation['width'] * slot_resize_scale,
                                          slot_annotation['height'] * slot_resize_scale)
            ann2_x = ann1_x + ann_w
            ann2_y = ann1_y + ann_h
            slot_mul = 1 / 4.0
            slot_1 = tform_res((ann1_x, ann1_y))[0]
            slot_2 = tform_res((ann2_x, ann2_y))[0]

            slot_bucket1_x = find_bucket(target_w, buckets, slot_1[0])
            slot_bucket1_y = find_bucket(target_h, buckets, slot_1[1])
            slot_bucket2_x = find_bucket(target_w, buckets, slot_2[0])
            slot_bucket2_y = find_bucket(target_h, buckets, slot_2[1])
            #print slot_1, slot_2
            #print 'slot1_bucket', slot_bucket1_x, slot_bucket1_y
            #print 'slot2_bucket', slot_bucket2_x, slot_bucket2_y
            go_indygo('slot_point1_x', slot_bucket1_x, slot_bucket1_y, slot_mul)
            go_indygo('slot_point1_y', slot_bucket1_y, slot_bucket1_x, slot_mul)
            go_indygo('slot_point2_x', slot_bucket2_x, slot_bucket2_y, slot_mul)
            go_indygo('slot_point2_y', slot_bucket2_y, slot_bucket2_x, slot_mul)

            # indygo
            indygo_resize_scale = 0.25
            point_1 = recipe.annotations['point1'][0]
            point_2 = recipe.annotations['point2'][0]
            point_1['x'] *= indygo_resize_scale
            point_1['y'] *= indygo_resize_scale


            point_2['x'] *= indygo_resize_scale
            point_2['y'] *= indygo_resize_scale

            indygo_mul = 1 / 4.0
            indygo_res1 = tform_res((point_1['x'], point_1['y']))[0]
            indygo_res2 = tform_res((point_2['x'], point_2['y']))[0]
            indygo_bucket1_x = find_bucket(target_w, buckets, indygo_res1[0])
            indygo_bucket1_y = find_bucket(target_h, buckets, indygo_res1[1])
            indygo_bucket2_x = find_bucket(target_w, buckets, indygo_res2[0])
            indygo_bucket2_y = find_bucket(target_h, buckets, indygo_res2[1])
            go_indygo('indygo_point1_x', indygo_bucket1_x, indygo_bucket1_y, indygo_mul)
            go_indygo('indygo_point1_y', indygo_bucket1_y, indygo_bucket1_x, indygo_mul)
            go_indygo('indygo_point2_x', indygo_bucket2_x, indygo_bucket2_y, indygo_mul)
            go_indygo('indygo_point2_y', indygo_bucket2_y, indygo_bucket2_x, indygo_mul)
            ##


        info = {'tform_res': tform_res,
                'r': r
                }

        #print 'img_shape', img.shape

        return Bunch(x=img, y=true_dist, recipe=recipe, info=info)
    except Exception as e:
        print traceback.format_exc()
        raise


def fetch_rob_crop_recipe_historical(recipe, global_spec):
    try:
        print 'gooo'
        img = fetch_path_local(recipe.path)
        img_h = img.shape[0]
        img_w = img.shape[1]
        pre_h, pre_w = 256, 256
        target_h, target_w = global_spec['target_h'], global_spec['target_w']
        TARGETS = global_spec['TARGETS']
        buckets = global_spec['buckets']

        sx = (pre_w - target_w + 1) / 2
        sy = (pre_h - target_h + 1) / 2
        #print sx, sy

        #target_h, target_w = 256, 256
        #print 'img_size', img.shape
        true_dist = np.zeros(shape=(WhaleTrainer.get_y_shape(TARGETS),), dtype=floatX)

        tform_res = AffineTransform(scale=(pre_w / float(img_w), pre_h / float(img_h)))
        tform_res += SimilarityTransform(translation=(-sx, -sy))

        tform_augment, r = unpack(random_perturbation_transform(rng=np.random, **global_spec['augmentation_params']),
                                  'tform', 'r')

        tform_center, tform_uncenter = build_center_uncenter_transforms2(target_w, target_h)
        tform_res += tform_center + tform_augment + tform_uncenter

        img = fast_warp(img, AffineTransform(tform_res._inv_matrix), output_shape=(target_h, target_w))
        img = transformation_historical(img, global_spec, perturb=False)

        if recipe.fetch_true_dist:
            # Constructing true_dist

            def go_indygo(name, a, b, v):
                inter = WhaleTrainer.get_interval(name, TARGETS)
                if a != -1 and b != -1 and inter is not None:
                    idx = inter[0] + a
                    if idx < inter[1]:
                        true_dist[idx] = v

            # slot
            slot_resize_scale = 0.25
            slot_annotation = recipe.annotations['slot'][0]
            ann1_x, ann1_y, ann_w, ann_h = (slot_annotation['x'] * slot_resize_scale,
                                          slot_annotation['y'] * slot_resize_scale,
                                          slot_annotation['width'] * slot_resize_scale,
                                          slot_annotation['height'] * slot_resize_scale)
            ann2_x = ann1_x + ann_w
            ann2_y = ann1_y + ann_h
            slot_mul = 1 / 4.0
            slot_1 = tform_res((ann1_x, ann1_y))[0]
            slot_2 = tform_res((ann2_x, ann2_y))[0]

            slot_bucket1_x = find_bucket(target_w, buckets, slot_1[0])
            slot_bucket1_y = find_bucket(target_h, buckets, slot_1[1])
            slot_bucket2_x = find_bucket(target_w, buckets, slot_2[0])
            slot_bucket2_y = find_bucket(target_h, buckets, slot_2[1])
            #print slot_1, slot_2
            #print 'slot1_bucket', slot_bucket1_x, slot_bucket1_y
            #print 'slot2_bucket', slot_bucket2_x, slot_bucket2_y
            go_indygo('slot_point1_x', slot_bucket1_x, slot_bucket1_y, slot_mul)
            go_indygo('slot_point1_y', slot_bucket1_y, slot_bucket1_x, slot_mul)
            go_indygo('slot_point2_x', slot_bucket2_x, slot_bucket2_y, slot_mul)
            go_indygo('slot_point2_y', slot_bucket2_y, slot_bucket2_x, slot_mul)

            # indygo
            indygo_resize_scale = 0.25
            point_1 = recipe.annotations['point1'][0]
            point_2 = recipe.annotations['point2'][0]
            point_1['x'] *= indygo_resize_scale
            point_1['y'] *= indygo_resize_scale


            point_2['x'] *= indygo_resize_scale
            point_2['y'] *= indygo_resize_scale

            indygo_mul = 1 / 4.0
            indygo_res1 = tform_res((point_1['x'], point_1['y']))[0]
            indygo_res2 = tform_res((point_2['x'], point_2['y']))[0]
            indygo_bucket1_x = find_bucket(target_w, buckets, indygo_res1[0])
            indygo_bucket1_y = find_bucket(target_h, buckets, indygo_res1[1])
            indygo_bucket2_x = find_bucket(target_w, buckets, indygo_res2[0])
            indygo_bucket2_y = find_bucket(target_h, buckets, indygo_res2[1])
            go_indygo('indygo_point1_x', indygo_bucket1_x, indygo_bucket1_y, indygo_mul)
            go_indygo('indygo_point1_y', indygo_bucket1_y, indygo_bucket1_x, indygo_mul)
            go_indygo('indygo_point2_x', indygo_bucket2_x, indygo_bucket2_y, indygo_mul)
            go_indygo('indygo_point2_y', indygo_bucket2_y, indygo_bucket2_x, indygo_mul)
            ##


        info = {'tform_res': tform_res,
                'r': r
                }

        #print 'img_shape', img.shape

        return Bunch(x=img, y=true_dist, recipe=recipe, info=info)
    except Exception as e:
        print traceback.format_exc()
        raise
