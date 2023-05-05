#导入所需的包
import cv2
import numpy as np
from PIL import Image
import math
from sympy import *

#定义展示图片的函数cv_show
def cv_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#定义函数遍历图像中的每个像素点，灰度不为0的像素就置为1，否则为0
def pretreatment(ima):
    ima = ima.convert('L')
    im = np.array(ima)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i, j] > 5:
                im[i, j] = 1
            else:
                im[i, j] = 0
    return im


#读取图像并转化为灰度图
original_img = cv2.imread('E:\jupyter\opencv\project\Extraction of PVP parameters\original_images\example_2.jpg')
cv_show('original_img', original_img)

#original_img_cut = original_img[150:330, 70:320]
original_img_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
#cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\cut_images\original_img_cut.png', original_img_cut)
#cv_show('gray', original_img_gray)
#掩模
original_mask = np.zeros([512, 512], dtype=np.uint8)
original_mask[100:380, 30:450] = 255
#cv_show('mask', original_mask)
original_roi = cv2.add(original_img_gray, np.zeros(np.shape(original_img_gray), dtype=np.uint8), mask=original_mask)
#cv_show('roi', original_roi)
#二值化处理，消除非骨质区域,使用滑动条调整二值化的阈值
ret, Outer_contour_thresh = cv2.threshold(original_roi, 160, 255, cv2.THRESH_BINARY)
#cv_show('thresh', Outer_contour_thresh)
# 中值滤波
median = cv2.medianBlur(Outer_contour_thresh, 5)
#blur = cv2.blur(Outer_contour_thresh, (3, 3))
#cv_show('median', median)
#提取轮廓信息
original_binary, original_contours, original_hierarchy = cv2.findContours(median, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#画出轮廓
original_draw_img = original_img.copy()
Outer_contour = cv2.drawContours(original_draw_img, original_contours, -1, (0, 0, 255), 1)
#cv_show('pictures', Outer_contour)


# 计算轮廓周长并排序
original_perimeter_list = []
for original_cnt in original_contours:
    original_perimeter = cv2.arcLength(original_cnt, True)
    original_perimeter_list.append(original_perimeter)
original_perimeter_list.sort(reverse=True)
print(original_perimeter_list)
#筛选出轮廓周长最长的两个轮廓
original_img_contours = []
for i in range(len(original_contours)):
    original_img_temp = np.zeros(original_img.shape, np.uint8)
    original_img_contours.append(original_img_temp)
    original_length = cv2.arcLength(original_contours[i], True)
    #设定保留轮廓的条件为轮廓周长大于200
    if original_length > original_perimeter_list[0] - 2:
         #print("轮廓 %d 的周长是:%d" % (i, length))
         cv2.drawContours(original_img_contours[i], original_contours, i, (0, 255, 0), 1)
         cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\outer_contours\outer_contour_0.png ', original_img_contours[i])
         #cv_show("轮廓 %d" % i, original_img_contours[i])
    elif original_perimeter_list[0] > original_length > original_perimeter_list[1] - 2:
         #print("轮廓 %d 的周长是:%d" % (i, original_length))
         cv2.drawContours(original_img_contours[i], original_contours, i, (0, 255, 0), 1)
         cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\inner_contours\inner_contour_0.png ',original_img_contours[i])
         cv_show("轮廓 %d" % i, original_img_contours[i])


#导入图像并转换我数组形式保存
inner_contour_0_ima = Image.open('E:\jupyter\opencv\project\Extraction of PVP parameters\inner_contours\inner_contour_0.png')
im_inner_contour_0 = pretreatment(inner_contour_0_ima)
index_inner_contour_0 = []
for i in range(im_inner_contour_0.shape[0]):
    for j in range(im_inner_contour_0.shape[1]):
        if im_inner_contour_0[i, j] == 1:
            index_inner_contour_0.append([i, j])
#print(index_inner_contour_0)

#寻找椎体内边界起始点坐标
starting_point = []
i = 0
count = 1
while i < len(index_inner_contour_0) - 5:
    if index_inner_contour_0[i][0] - 1 == index_inner_contour_0[i - 1][0]:
        count += 1
        a = 0
        while index_inner_contour_0[i + a + 1][0] == index_inner_contour_0[i][0]:
            k = 1
            h = 1
            for j in range(count):
                if abs(index_inner_contour_0[i + a][1] - index_inner_contour_0[i - 1 - j][1]) < 10:
                    k = 0
                else:
                    h = 1
                if k and h == 0:
                    break
            if k and h == 1:
                #print(index_inner_contour_0[i + a])
                starting_point.append(index_inner_contour_0[i + a])
                break

            a += 1
            #print(a, "___", i)
        count = 1
    else:
        count += 1
    i += 1
#print(starting_point)


#连接两个椎体内边界起始点
inner_contour_0 = cv2.imread('E:\jupyter\opencv\project\Extraction of PVP parameters\inner_contours\inner_contour_0.png')
#cv_show('inner_contour_0', inner_contour_0)
inner_contour_0_connect = cv2.line(inner_contour_0, (starting_point[0][1], starting_point[0][0]), (starting_point[1][1], starting_point[1][0]), (0, 0, 255), 1)
#cv_show('connect', inner_contour_0_connect)
#再次提取轮廓
connect_gray = cv2.cvtColor(inner_contour_0_connect, cv2.COLOR_BGR2GRAY)
ret, connect_thresh = cv2.threshold(connect_gray, 20, 255, cv2.THRESH_BINARY)
connect_binary, connect_contours, connect_hierarchy = cv2.findContours(connect_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
connect_draw = inner_contour_0_connect.copy()
connect_pictures = cv2.drawContours(connect_draw, connect_contours, -1, (0, 255, 0), 1)
#cv_show('connect_pictures', connect_pictures)

#计算轮廓周长
connect_perimeter_list = []
for cnt in connect_contours:
    perimeter = cv2.arcLength(cnt, True)
    connect_perimeter_list.append(perimeter)

connect_perimeter_list.sort(reverse=True)
#print(connect_perimeter_list)

#筛选出轮廓周长较长的轮廓，以此过滤掉噪声轮廓
inner_connect_contours = []
for i in range(len(connect_contours)):
    connect_img_temp = np.zeros(inner_contour_0_connect.shape, np.uint8)
    inner_connect_contours.append(connect_img_temp)
    connect_length = cv2.arcLength(connect_contours[i], True)
    #设定保留轮廓的条件为轮廓周长小于240
    if connect_perimeter_list[1] > connect_length > connect_perimeter_list[2] - 2:
         #print("轮廓 %d 的周长是:%d" % (i, connect_length))
         cv2.drawContours(inner_connect_contours[i], connect_contours, i, (0, 255, 0), 1)
         cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\inner_contours\inner_contour_1.png ', inner_connect_contours[i])
         #cv_show("轮廓 %d" % i, inner_connect_contours[i])
outer_contour_0 = cv2.imread('E:\jupyter\opencv\project\Extraction of PVP parameters\outer_contours\outer_contour_0.png')
inner_contour_1 = cv2.imread('E:\jupyter\opencv\project\Extraction of PVP parameters\inner_contours\inner_contour_1.png')
centrum = cv2.addWeighted(outer_contour_0, 1, inner_contour_1, 1, 0)
#cv_show('centrum', centrum)
cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\centrums\centrum.png', centrum)

#皮肤边界草图提取
skin_mask = np.zeros([512, 512], dtype=np.uint8)
skin_mask[340:380, :] = 255
#cv_show('mask', skin_mask)
skin_roi = cv2.add(original_img_gray, np.zeros(np.shape(original_img_gray), dtype=np.uint8), mask=skin_mask)
#cv_show('roi', skin_roi)
ret, skin_thresh = cv2.threshold(skin_roi, 48, 255, cv2.THRESH_BINARY)
#cv_show('thresh', skin_thresh)
skin_binary, skin_contours, skin_hierarchy = cv2.findContours(skin_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
skin_draw_img = original_img.copy()
skin_pictures = cv2.drawContours(skin_draw_img, skin_contours, -1, (0, 0, 255), 1)
#cv_show('pictures', skin_pictures)
# 计算轮廓周长
skin_perimeter_list = []
for skin_cnt in skin_contours:
    skin_perimeter = cv2.arcLength(skin_cnt, True)
    skin_perimeter_list.append(skin_perimeter)
skin_perimeter_list.sort(reverse=True)
    #print(skin_perimeter)
#筛选出轮廓周长较长的轮廓，以此过滤掉噪声轮廓
skin_contours_len = []
for i in range(len(skin_contours)):
    skin_img_temp = np.zeros(original_img.shape, np.uint8)
    skin_contours_len.append(skin_img_temp)
    skin_length = cv2.arcLength(skin_contours[i], True)
    #设定保留轮廓的条件为轮廓周长大于500
    if skin_length > skin_perimeter_list[0] - 2:
         #print("轮廓 %d 的周长是:%d" % (i, skin_length))
         cv2.drawContours(skin_contours_len[i], skin_contours, i, (0, 255, 0), 1)
         cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\skin_contours\original_skin_contour.png', skin_contours_len[i])
         #cv_show("轮廓 %d" % i, skin_contours_len[i])

#获取所需的皮肤部分边界
#导入图像并转换我数组形式保存
finall_skin = cv2.imread('E:\jupyter\opencv\project\Extraction of PVP parameters\skin_contours\original_skin_contour.png')
#cv_show('finall_skin', finall_skin)
finall_skin_ima = Image.open('E:\jupyter\opencv\project\Extraction of PVP parameters\skin_contours\original_skin_contour.png')
finall_skin_im = pretreatment(finall_skin_ima)

finall_skin_index = []
for i in range(finall_skin_im.shape[1]):
    for j in range(finall_skin_im.shape[0]):
        if finall_skin_im[j, i] == 1:
            finall_skin_index.append([j, i])
s = 0
while s < len(finall_skin_index) - 2:
    if finall_skin_index[s][1] == finall_skin_index[s + 1][1]:
        finall_skin[finall_skin_index[s][0], finall_skin_index[s][1]] = [0, 0, 0]
    s += 1
#cv_show('finall_skin', finall_skin)
cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\skin_contours\\finall_skin.png', finall_skin)

#组合分割结果
centrums_1 = cv2.imread('E:\jupyter\opencv\project\Extraction of PVP parameters\centrums\centrum.png')
finall_skin_1 = cv2.imread('E:\jupyter\opencv\project\Extraction of PVP parameters\skin_contours\\finall_skin.png')
divide_result = cv2.addWeighted(centrums_1, 1, finall_skin_1, 1, 0)
#cv_show('divide_result', divide_result)
cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\divide_results\divide_result.png', divide_result)

#显示分割效果
divide_result_1 = cv2.imread('E:\jupyter\opencv\project\Extraction of PVP parameters\divide_results\divide_result.png')
original_image_1 = cv2.imread('E:\jupyter\opencv\project\Extraction of PVP parameters\original_images\example_1.jpg')
merge = cv2.addWeighted(divide_result_1, 1, original_image_1, 0.7, 0)
#cv_show('merge', merge)
cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\merge_images\merge.png', merge)


#参数提取

#寻找椎体”中心“（内轮廓所有像素点的平均坐标）
para_inner_contour = cv2.imread('E:\jupyter\opencv\project\Extraction of PVP parameters\inner_contours\inner_contour_1.png')
#cv_show('para_inner_contour', para_inner_contour)
para_inner_ima = Image.open('E:\jupyter\opencv\project\Extraction of PVP parameters\inner_contours\inner_contour_1.png')
para_inner_im = pretreatment(para_inner_ima)
para_inner_index = []
for i in range(para_inner_im.shape[0]):
    for j in range(para_inner_im.shape[1]):
        if para_inner_im[i, j] == 1:
            para_inner_index.append([i, j])
#计算内轮廓的平均像素点坐标值，确定为中心位置
i = 0
sum_1 = 0
sum_2 = 0
while i <= len(para_inner_index) - 1:
    x = para_inner_index[i][0]
    y = para_inner_index[i][1]
    sum_1 += x
    sum_2 += y
    i += 1
average_1 = sum_1 // i
average_2 = sum_2 // i
#print(average_1)
#print(average_2)
#把中心点画成红色
divide_result_1[average_1, average_2] = [0, 0, 255]
#cv_show('divide_result_1', divide_result_1)
cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\divide_results\divide_result_1.png', divide_result_1)

#计算椎体内边界底角点
end_sum = 0
end_count = 0
for i in range(len(para_inner_index)):
    if para_inner_index[i][0] == para_inner_index[-1][0]:
        end_sum = end_sum + para_inner_index[i][1]
        end_count += 1
    else:
        continue
end_avg = end_sum // end_count
end_point = [para_inner_index[-1][0], end_avg]

#图像旋转
#rows, cols = divide_result_1.shape[:2]
#M = cv2.getRotationMatrix2D((average_2, average_1), 6, 1)
#spin_1 = cv2.warpAffine(divide_result_1, M, (cols, rows))
#cv_show('spin_1', spin_1)
#cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\spin_images\spin_1.png', spin_1)
#提取椎体边界坐标信息
top_ima = Image.open('E:\jupyter\opencv\project\Extraction of PVP parameters\divide_results\divide_result_1.png')
top_im = pretreatment(top_ima)
#打印出像素为1的像素点的索引
top_index = []
for i in range(top_im.shape[0]):
    for j in range(top_im.shape[1]):
        if top_im[i, j] == 1:
            top_index.append([i, j])

#通过椎体中心做一条起始线
fake_distance_list = []
intersection_point_1_list = []
intersection_point_2_list = []
count = -1
for k in range(30):
    spin_2 = divide_result_1.copy()
    cv2.line(spin_2, (0, average_1-8*k + 30), (average_2, average_1), (0, 0, 255), 1)
    #cv_show('spin_2', spin_2)
    #找出交点，并计算两交点之间的距离，如果满足某条件，跳出循环
    mask_1 = np.zeros(spin_2.shape, np.uint8)
    cv2.line(mask_1, (0, average_1-8*k + 30), (average_2, average_1), (255, 255, 255), 1)
    #cv_show('mask', mask_1)
    cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\mask_images\mask_1.png', mask_1)
    mask_1_ima = Image.open('E:\jupyter\opencv\project\Extraction of PVP parameters\mask_images\mask_1.png')
    mask_1_im = pretreatment(mask_1_ima)
    centrum_points = []
    for i in range(mask_1.shape[1]):
        for j in range(mask_1.shape[0]):
            if top_im[j, i] and mask_1_im[j, i] == 1:
                centrum_points.append([j, i])
    #print(centrum_points)
    #print(len(centrum_points))
    if len(centrum_points) >= 3:
        count += 1
        intersection_point_1 = centrum_points[0]
        intersection_point_1_list.append(intersection_point_1)
        #print(intersection_point_1)
        intersection_point_2 = centrum_points[len(centrum_points)-2]
        intersection_point_2_list.append(intersection_point_2)
        #print(intersection_point_2)
        fake_distance = math.sqrt(((intersection_point_1[0] - intersection_point_2[0]) ** 2) + ((intersection_point_1[1] - intersection_point_2[1]) ** 2))
        #print(fake_distance)
        fake_distance_list.append(fake_distance)
        if k >= 1:
            if fake_distance_list[count] <= fake_distance_list[count-1]:
                continue
            else:
                mid_point = [((intersection_point_1_list[count-1][0] + intersection_point_2_list[count-1][0])) // 2 , ((intersection_point_1_list[count-1][1] + intersection_point_2_list[count-1][1])) // 2]
                #print(intersection_point_1)
                #print(intersection_point_2)
                #print(k)
                #print(fake_distance_list[k-1])
                #print(mid_point)
                break

#通过关键点画线，获取直线表达式
spin_3 = divide_result_1.copy()

cv2.line(spin_3, (end_point[1], end_point[0]), (average_2, average_1), (0, 0, 255), 1)
slope_1 = ((-average_1) - (-end_point[0])) / (average_2 - end_point[1] + 0.01)
b1 = -average_1 - slope_1 * average_2
pre_end_point_1 = (0 - b1) // slope_1
pre_end_point_2 = (-512 - b1) // slope_1
end_point_1 = int(pre_end_point_1)
end_point_2 = int(pre_end_point_2)
#画出结果
spin_4 = divide_result_1.copy()
cv2.line(spin_4, (end_point_1, 0), (end_point_2, 512), (0, 0, 255), 1)

# 寻找两直线与皮肤边界的交点
mask_2 = np.zeros(spin_4.shape, np.uint8)
cv2.line(mask_2, (end_point_1, 0), (end_point_2, 512), (0, 0, 255), 1)
#cv_show('mask', mask_2)
cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\mask_images\mask_2.png', mask_2)
mask_2_ima = Image.open('E:\jupyter\opencv\project\Extraction of PVP parameters\mask_images\mask_2.png')
mask_2_im = pretreatment(mask_2_ima)
skin_inter_points_list1 = []
for i in range(mask_2.shape[0]):
    for j in range(mask_2.shape[1]):
        if top_im[i, j] and mask_2_im[i, j] == 1:
            skin_inter_points_list1.append([i, j])
    # print(point)
    # print(len(point))

top = skin_inter_points_list1[0]
skin_intersection_point_1 = skin_inter_points_list1[-1]

#画左边第一条线
cv2.line(spin_3, (top[1], top[0]), (mid_point[1], mid_point[0]), (0, 0, 255), 1)
slope_2 = ((-mid_point[0]) - (-top[0])) / (mid_point[1] - top[1])
b2 = -mid_point[0] - slope_2 * mid_point[1]
pre_end_point_3 = (-512 - b2) // slope_2
end_point_3 = int(pre_end_point_3)
cv2.line(spin_4, (top[1], top[0]), (end_point_3, 512), (0, 0, 255), 1)
#cv_show('spin_4', spin_4)

mask_3 = np.zeros(spin_4.shape, np.uint8)
cv2.line(mask_3, (top[1], top[0]), (end_point_3, 512), (255, 255, 255), 1)
#cv_show('mask', mask_3)
cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\mask_images\mask_3.png', mask_3)
mask_3_ima = Image.open('E:\jupyter\opencv\project\Extraction of PVP parameters\mask_images\mask_3.png')
mask_3_im = pretreatment(mask_3_ima)
skin_inter_points_list2 = []
for i in range(mask_3.shape[0]):
    for j in range(mask_3.shape[1]):
        if top_im[i, j] and mask_3_im[i, j] == 1:
            skin_inter_points_list1.append([i, j])
    # print(point)
    # print(len(point))
skin_intersection_point_2 = skin_inter_points_list1[-1]
skin_intersection_point_3 = skin_inter_points_list1[-2]
# intersection_point_1_list.append(intersection_point_1)
#print(skin_intersection_point_2)


#通过交点1向直线2做水平线，并导出交点，用以获得旁开棘突距离
slope_3 = -1 / slope_1
#print(slope_3)
b3 = (-skin_intersection_point_2[0]) - (slope_3 * skin_intersection_point_2[1])
#print(b3)
level_x = symbols('level_x')
level_x_result = solve((slope_3 - slope_1) * level_x + b3 - b1, level_x)
#print(level_x_result)
level_x = int(level_x_result[0]) + 1
#print(level_x)
level_y = -(slope_3 * level_x + b3)
level_y = int(level_y)
level_point = [level_y, level_x]
#print(level_point)

b4 = -skin_intersection_point_3[0] - slope_1 * skin_intersection_point_3[1]
pre_end_point_4 = (-512 - b4) // slope_1
end_point_4 = int(pre_end_point_4)
cv2.line(spin_4, (skin_intersection_point_3[1], skin_intersection_point_3[0]), (end_point_4, 512), (0, 0, 255), 1)

mask_4 = np.zeros(spin_4.shape, np.uint8)
cv2.line(mask_4, (skin_intersection_point_3[1], skin_intersection_point_3[0]), (end_point_4, 512), (255, 255, 255), 1)
#cv_show('mask', mask_4)
cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\mask_images\mask_4.png', mask_4)
mask_4_ima = Image.open('E:\jupyter\opencv\project\Extraction of PVP parameters\mask_images\mask_4.png')
mask_4_im = pretreatment(mask_4_ima)
skin_inter_points_list3 = []
for i in range(mask_4.shape[0]):
    for j in range(mask_3.shape[1]):
        if top_im[i, j] and mask_4_im[i, j] == 1:
            skin_inter_points_list3.append([i, j])
    # print(point)
    # print(len(point))
skin_intersection_point_4 = skin_inter_points_list1[-1]

#绘制结果
spin_5 = cv2.line(spin_4, (skin_intersection_point_2[1], skin_intersection_point_2[0]), (level_point[1], level_point[0]), (0, 0, 255), 1)
#cv_show('spin_4', spin_4)
cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\spin_images\spin_5.png', spin_5)

#将结果与原图融合
target = cv2.imread('E:\jupyter\opencv\project\Extraction of PVP parameters\spin_images\spin_5.png')
#rows, cols = target.shape[:2]
#M = cv2.getRotationMatrix2D((average_2, average_1), -6, 1)
#target_1 = cv2.warpAffine(target, M, (cols, rows))
finall = cv2.addWeighted(target, 1, original_img, 0.7, 0)
cv_show('finall', finall)
cv2.imwrite('finall.png', finall)

#参数计算过程
parameter_AC = math.sqrt((abs(top[1] - skin_intersection_point_2[1]) ** 2) + (abs(top[0] - skin_intersection_point_2[0]) ** 2))
parameter_AD = math.sqrt((abs(top[1] - skin_intersection_point_1[1]) ** 2) + (abs(top[0] - skin_intersection_point_1[0]) ** 2))
parameter_AE = math.sqrt((abs(top[1] - level_point[1]) ** 2) + (abs(top[0] - level_point[0]) ** 2))
parameter_CE = math.sqrt((abs(skin_intersection_point_2[1] - level_point[1]) ** 2) + (abs(skin_intersection_point_2[0] - level_point[0]) ** 2))
parameter_FG = math.sqrt((abs(skin_intersection_point_3[1] - skin_intersection_point_4[1]) ** 2) + (abs(skin_intersection_point_3[0] - skin_intersection_point_4[0]) ** 2))
parameter_CG = parameter_CE * (parameter_FG / parameter_AE)
parameter_GE = parameter_CE - parameter_CG
parameter_Angle = math.atan(math.sqrt(parameter_CE / parameter_AE))*180/math.pi
#print('%.2f' % parameter_AC)
#print('%.2f' % parameter_AD)
#print('%.2f' % parameter_AE)
#print('%.2f' % parameter_CE)
#print('%.2f' % parameter_Angle)
#实际距离转化（转化方法有待考量）
rate = 0.2734375
AC = parameter_AC * rate
AD = parameter_AD * rate
AE = parameter_AE * rate
CE = parameter_CE * rate
FG = parameter_FG * rate
CG = parameter_CG * rate
GE = parameter_GE * rate
print("AC = ", '%.2f' % AC, "mm")
print("AD = ", '%.2f' % AD, "mm")
print("AE = ", '%.2f' % AE, "mm")
print("CE = ", '%.2f' % CE, "mm")
print("FG = ", '%.2f' % FG, "mm")
print("CG = ", '%.2f' % CG, "mm")
print("GE = ", '%.2f' % GE, "mm")
print("Angle = ", '%.2f' % parameter_Angle, "°")
#图像尺寸大小与人体实际大小的转化
