import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as sc


def feature_detection(path, height): # step 1

    img_bgr = cv2.imread(path)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (img_gray.shape[1], height))

    sift = cv2.SIFT_create()


    pts, des = sift.detectAndCompute(img_gray, None)

    output = cv2.drawKeypoints(img_gray, pts, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.figure(figsize=(10, 4))


    plt.subplot(1, 2, 1)
    plt.title('gray_img')
    plt.imshow(img_gray,cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('features')
    plt.imshow(output)
    plt.axis('off')

    plt.savefig('./yosemite/step1.png',dpi = 300)
    plt.show()

    return

def warpImages_add(img1, img2):


    sift = cv2.SIFT_create()
    pts1, des1 = sift.detectAndCompute(img1, None)
    pts2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches_ = bf.match(des2, des1)

    matches_ = sorted(matches_, key=lambda x: x.distance)

    query_pts = np.float32([pts2[m.queryIdx].pt for m in matches_[:10]]).reshape(-1, 1, 2)
    train_pts = np.float32([pts1[m.trainIdx].pt for m in matches_[:10]]).reshape(-1, 1, 2)
    matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)

    corr_match_img = cv2.drawMatches(img2, pts2, img1, pts1, matches_[:10], None, flags=2)
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]


    list_of_points_1 = np.float32([[0,0], [0,rows1], [cols1,rows1], [cols1,0]]).reshape(-1,1,2)
    temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)
    list_of_points_2 = cv2.perspectiveTransform(temp_points, matrix)
    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]])

    output_img = np.float32(cv2.warpPerspective(img2, H_translation.dot(matrix), (x_max-x_min, y_max-y_min)))


    output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] += np.float32(img1)

    output_img = output_img / 2

    return output_img.astype(np.uint8),corr_match_img


def warpImages(img1, img2):
    sift = cv2.SIFT_create()
    pts1, des1 = sift.detectAndCompute(img1, None)
    pts2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches_ = bf.match(des2, des1)


    matches_ = sorted(matches_, key=lambda x: x.distance)

    query_pts = np.float32([pts2[m.queryIdx].pt for m in matches_[:10]]).reshape(-1, 1, 2)
    train_pts = np.float32([pts1[m.trainIdx].pt for m in matches_[:10]]).reshape(-1, 1, 2)
    matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0,0], [0,rows1], [cols1,rows1], [cols1,0]]).reshape(-1,1,2)
    temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)
    list_of_points_2 = cv2.perspectiveTransform(temp_points, matrix)
    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
    translation_dist = [-x_min, -y_min]

    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]])

    output_img = cv2.warpPerspective(img2, H_translation.dot(matrix), (x_max-x_min, y_max-y_min))
    output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1

    return output_img

def get_distance_transform(img_rgb):

    thresh = cv2.threshold(img_rgb,0,255,cv2.THRESH_BINARY)[1]

    thresh = thresh.any(axis = 2)

    thresh = np.pad(thresh,1)

    dist = sc.distance_transform_edt(thresh)[1:-1,1:-1]

    dist = dist [: , : , None ]

    return np.float32(dist / dist.max())

def dnorm_warpImages(left_img, right_img):
    sift = cv2.SIFT_create()
    pts1, des1 = sift.detectAndCompute(left_img, None)
    pts2, des2 = sift.detectAndCompute(right_img, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches_ = bf.match(des2, des1)

    matches_ = sorted(matches_, key=lambda x: x.distance)
    query_pts = np.float32([pts2[m.queryIdx].pt for m in matches_[:10]]).reshape(-1, 1, 2)
    train_pts = np.float32([pts1[m.trainIdx].pt for m in matches_[:10]]).reshape(-1, 1, 2)
    matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
    rows1, cols1 = left_img.shape[:2]
    rows2, cols2 = right_img.shape[:2]

    list_of_points_1 = np.float32([[0,0], [0,rows1], [cols1,rows1], [cols1,0]]).reshape(-1,1,2)
    temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)
    list_of_points_2 = cv2.perspectiveTransform(temp_points, matrix)
    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]

    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]])

    intersect_axis = np.int32(np.array(list_of_points_2).min(axis=0))[0][0]

    alpha = 0.6
    start_point = np.int32(alpha * (cols1 - intersect_axis) + intersect_axis)
    warp_img = np.float32(cv2.warpPerspective(right_img, H_translation.dot(matrix), (x_max-x_min, y_max-y_min)))
    #output_img = warp_img.copy()
    output_img = np.zeros_like(warp_img,dtype=np.float32)
    output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0],:] = left_img


    dnorm = get_distance_transform(output_img)

    whole_area_denorm = np.zeros_like(warp_img)

    whole_area_denorm += dnorm

    #dnorm[:,:start_point,:] = 1.0
    whole_area_denorm[:,:start_point,:] = 1.0


    blend_img = whole_area_denorm * output_img + (1-whole_area_denorm) * warp_img


    #blend_img[:,:281,:] =  output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0],:]
    return np.clip(blend_img,0,255).astype(right_img.dtype)

def img_stitch(imgs):

    imgs_list = [[] for _ in range(len(imgs))]

    imgs_list[0] = imgs
    for i in range(1,len(imgs_list)):
        imgs_num = len(imgs_list[i-1])
        for j in range(1,imgs_num):
            img1 = imgs_list[i-1][j-1]
            img2 = imgs_list[i-1][j]

            img_stitch = dnorm_warpImages(img1,img2)
            imgs_list[i].append(img_stitch)


    last_result = imgs_list[len(imgs_list)-1][0]
    return last_result,imgs_list


def image_reszie(img,height, width):

    norm_img = cv2.resize(img, (width,height))
    return norm_img


if __name__ == '__main__':
    feature_detection('./yosemite/yosemite2.jpg',480)
    yos1_bgr = cv2.imread('./yosemite/yosemite1.jpg')
    yos1_rgb = cv2.cvtColor(yos1_bgr, cv2.COLOR_BGR2RGB)
    yos2_bgr = cv2.imread('./yosemite/yosemite2.jpg')
    yos2_rgb = cv2.cvtColor(yos2_bgr, cv2.COLOR_BGR2RGB)
    result,corr_match= warpImages_add(yos1_rgb,yos2_rgb)

    plt.subplot(1, 2, 1)
    plt.title('aligned_image')
    plt.imshow(result)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('matches')
    plt.imshow(corr_match,cmap='gray')
    plt.axis('off')

    plt.savefig('./yosemite/step2.png',dpi = 300)
    plt.show()


    result_2 = warpImages(yos1_rgb, yos2_rgb)
    blend_img = dnorm_warpImages(yos1_rgb, yos2_rgb)
    plt.subplot(1, 2, 1)
    plt.title('original')
    plt.imshow(result_2)

    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('blend')
    plt.imshow(blend_img)
    plt.axis('off')
    plt.savefig('./yosemite/step3.png', dpi=300)
    plt.show()

    yos3_bgr = cv2.imread('./yosemite/yosemite3.jpg')
    yos3_rgb = cv2.cvtColor(yos3_bgr, cv2.COLOR_BGR2RGB)
    yos4_bgr = cv2.imread('./yosemite/yosemite4.jpg')
    yos4_rgb = cv2.cvtColor(yos4_bgr, cv2.COLOR_BGR2RGB)

    imgs = [yos1_rgb,yos2_rgb,yos3_rgb,yos4_rgb]

    result, _ = img_stitch(imgs)

    plt.figure(figsize=(16, 5))
    plt.title('images stitching')
    plt.imshow(result)
    plt.axis('off')
    plt.savefig('./yosemite/step4.png', dpi=300)
    plt.show()

    myview1_bgr = cv2.imread('./my_data/1.jpg')
    myview1_rgb =  cv2.cvtColor(myview1_bgr, cv2.COLOR_BGR2RGB)
    myview2_bgr = cv2.imread('./my_data/2.jpg')
    myview2_rgb =  cv2.cvtColor(myview2_bgr, cv2.COLOR_BGR2RGB)
    myview3_bgr = cv2.imread('./my_data/3.jpg')
    myview3_rgb =  cv2.cvtColor(myview3_bgr, cv2.COLOR_BGR2RGB)
    myview4_bgr = cv2.imread('./my_data/4.jpg')
    myview4_rgb =  cv2.cvtColor(myview3_bgr, cv2.COLOR_BGR2RGB)

    imgs = [myview1_rgb,myview2_rgb,myview3_rgb,myview4_rgb]
    height_normalize = 480
    width_normalize = 640
    norm_img_list = [image_reszie(myview_rgb,height_normalize,width_normalize)
                     for myview_rgb in imgs]

    result,_ = img_stitch(norm_img_list)
    plt.figure(figsize=(16,5))
    plt.imshow(result)
    plt.axis('off')
    plt.savefig('./my_data/result.png', dpi=300)
    plt.show()

    building1_bgr = cv2.imread('./my_data/building1.png')
    building1_rgb =  cv2.cvtColor(building1_bgr, cv2.COLOR_BGR2RGB)
    building2_bgr = cv2.imread('./my_data/building2.png')
    building2_rgb =  cv2.cvtColor(building2_bgr, cv2.COLOR_BGR2RGB)
    building3_bgr = cv2.imread('./my_data/building3.png')
    building3_rgb =  cv2.cvtColor(building3_bgr, cv2.COLOR_BGR2RGB)
    building4_bgr = cv2.imread('./my_data/building4.png')
    building4_rgb =  cv2.cvtColor(building4_bgr, cv2.COLOR_BGR2RGB)
    imgs = [building1_rgb,building2_rgb,building3_rgb,building4_rgb]
    result,_ = img_stitch(imgs)
    plt.figure(figsize=(16,5))
    plt.imshow(_[2][1])
    plt.axis('off')
    plt.savefig('./my_data/result_7.png', dpi=300)
    plt.show()