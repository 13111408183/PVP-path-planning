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

def getDist_P2L(PointP, Pointa, Pointb):
    """计算点到直线的距离
        PointP：定点坐标
        Pointa：直线a点坐标
        Pointb：直线b点坐标
    """
    # 求直线方程
    A = (-Pointb[0] - (-Pointa[0])) / (Pointb[1] - Pointa[1] + 0.001)
    B = -1
    C = -Pointb[0] - (A * Pointb[1])
    # 代入点到直线距离公式
    distance = abs(A * PointP[1] + B * (-PointP[0]) + C) / math.sqrt((A * A) + (B * B))
    return distance

#读取图像并转化为灰度图
original_img = cv2.imread('E:\jupyter\opencv\project\Extraction of PVP parameters\original_images\example_69.jpg')
cv_show('original_img', original_img)

#转化为灰度图像
original_img_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
#cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\cut_images\original_img_cut.png', original_img_cut)
#cv_show('gray', original_img_gray)

#掩模操作，用来消除掉患者的相关个人信息
original_mask = np.zeros([512, 512], dtype=np.uint8)
original_mask[100:505, 30:450] = 255
#cv_show('mask', original_mask)
original_roi = cv2.add(original_img_gray, np.zeros(np.shape(original_img_gray), dtype=np.uint8), mask=original_mask)
#cv_show('roi', original_roi)

#二值化处理，消除非骨质区域,使用滑动条调整二值化的阈值
#ret, Outer_contour_thresh = cv2.threshold(original_roi, 160, 255, cv2.THRESH_BINARY)
#cv_show('thresh', Outer_contour_thresh)

ret, median = cv2.threshold(original_roi, 185, 255, cv2.THRESH_BINARY)
cv_show('thresh', median)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
closing = cv2.morphologyEx(median, cv2.MORPH_CLOSE, kernel)
cv_show('closing', closing)
#进行一次开运算
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#median = cv2.erode(thresh, kernel)
#cv_show('opening', median)
# 中值滤波
#median = cv2.medianBlur(Outer_contour_thresh, 5)
#cv_show('median', median)

#提取椎体轮廓信息
original_binary, original_contours, original_hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#画出轮廓
#复制原图，以免直接对原图进行修改
original_draw_img = original_img.copy()
Outer_contour = cv2.drawContours(original_draw_img, original_contours, -1, (0, 0, 255), 1)
cv_show('pictures', Outer_contour)

# 计算椎体轮廓面积并从大到小排序
original_perimeter_list = []
for original_cnt in original_contours:
    original_perimeter = cv2.contourArea(original_cnt, False)
    if original_perimeter >= 1000:

        original_perimeter_list.append(original_perimeter)
original_perimeter_list.sort(reverse=True)
print(original_perimeter_list)

#筛选出轮廓面积最大的两个轮廓
original_img_contours = []
for i in range(len(original_contours)):
    original_img_temp = np.zeros(original_img.shape, np.uint8)
    original_img_contours.append(original_img_temp)
    original_area = cv2.contourArea(original_contours[i], False)

    if len(original_perimeter_list) >= 3:

        #设定保留轮廓的条件为保存最长的前两个
        if original_area > original_perimeter_list[0] - 2:
             #print("轮廓 %d 的面积是:%d" % (i, area))
             cv2.drawContours(original_img_contours[i], original_contours, i, (0, 255, 0), 1)
             cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\outer_contours\outer_contour_0.png ', original_img_contours[i])
             #cv_show("轮廓 %d" % i, original_img_contours[i])

        elif original_perimeter_list[1] > original_area > original_perimeter_list[-1] - 2:
             #print("轮廓 %d 的面积是:%d" % (i, original_area))
             cv2.drawContours(original_img_contours[i], original_contours, i, (0, 255, 0), 1)
             cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\inner_contours\inner_contour_0.png ',original_img_contours[i])
             cv_show("轮廓 %d" % i, original_img_contours[i])
    else:
        if original_area > original_perimeter_list[0] - 2:
             #print("轮廓 %d 的面积是:%d" % (i, area))
             cv2.drawContours(original_img_contours[i], original_contours, i, (0, 255, 0), 1)
             cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\outer_contours\outer_contour_0.png ', original_img_contours[i])
             #cv_show("轮廓 %d" % i, original_img_contours[i])
        elif original_perimeter_list[0] > original_area > original_perimeter_list[1] - 2:
             #print("轮廓 %d 的面积是:%d" % (i, original_area))
             cv2.drawContours(original_img_contours[i], original_contours, i, (0, 255, 0), 1)
             cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\inner_contours\inner_contour_0.png ',original_img_contours[i])
             cv_show("轮廓 %d" % i, original_img_contours[i])
             if original_perimeter_list[1] >= 3500:

                #导入内轮廓图像并转换为数组形式保存
                inner_contour_0_ima = Image.open('E:\jupyter\opencv\project\Extraction of PVP parameters\inner_contours\inner_contour_0.png')
                im_inner_contour_0 = pretreatment(inner_contour_0_ima)

                #把椎体内边界轮廓像素值不为0的像素点索引存储起来
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
                while i < len(index_inner_contour_0) - 10:
                    if index_inner_contour_0[i][0] - 1 == index_inner_contour_0[i - 1][0]:
                        count += 1
                        a = 0
                        while index_inner_contour_0[i + a + 1][0] == index_inner_contour_0[i][0]:
                            k = 1
                            h = 1
                            for j in range(count):
                                if abs(index_inner_contour_0[i + a][1] - index_inner_contour_0[i - 1 - j][1]) < 15:
                                    k = 0
                                else:
                                    h = 1
                                if k and h == 0:
                                    break
                            if k and h == 1:
                                #print(index_inner_contour_0[i + a])
                                starting_point.append(index_inner_contour_0[i + a])
                                #break

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
                inner_contour_0_connect = cv2.line(inner_contour_0, (starting_point[0][1], starting_point[0][0]), (starting_point[-1][1], starting_point[-1][0]), (0, 0, 255), 1)
                #cv_show('connect', inner_contour_0_connect)
                #再次提取轮廓并绘制
                connect_gray = cv2.cvtColor(inner_contour_0_connect, cv2.COLOR_BGR2GRAY)
                ret, connect_thresh = cv2.threshold(connect_gray, 20, 255, cv2.THRESH_BINARY)
                connect_binary, connect_contours, connect_hierarchy = cv2.findContours(connect_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                connect_draw = inner_contour_0_connect.copy()
                connect_pictures = cv2.drawContours(connect_draw, connect_contours, -1, (0, 255, 0), 1)
                cv_show('connect_pictures', connect_pictures)

                #计算轮廓周长
                connect_perimeter_list = []
                for cnt in connect_contours:
                    perimeter = cv2.arcLength(cnt, True)
                    connect_perimeter_list.append(perimeter)
                connect_perimeter_list.sort(reverse=True)
                #print(connect_perimeter_list)

                #根据周长筛选出所需要的内边界轮廓
                inner_connect_contours = []
                for i in range(len(connect_contours)):
                    connect_img_temp = np.zeros(inner_contour_0_connect.shape, np.uint8)
                    inner_connect_contours.append(connect_img_temp)
                    connect_length = cv2.arcLength(connect_contours[i], True)
                    #设定保留轮廓的条件为轮廓周长第3长的
                    if connect_perimeter_list[1] > connect_length > connect_perimeter_list[2] - 2:
                         #print("轮廓 %d 的周长是:%d" % (i, connect_length))
                         cv2.drawContours(inner_connect_contours[i], connect_contours, i, (0, 255, 0), 1)
                         cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\inner_contours\inner_contour_0.png ', inner_connect_contours[i])
                         cv_show("轮廓 %d" % i, inner_connect_contours[i])

                #outer_contour_0 = cv2.imread('E:\jupyter\opencv\project\Extraction of PVP parameters\outer_contours\outer_contour_0.png')
                #inner_contour_1 = cv2.imread('E:\jupyter\opencv\project\Extraction of PVP parameters\inner_contours\inner_contour_0.png')
                #centrum = cv2.addWeighted(outer_contour_0, 1, inner_contour_1, 1, 0)
                #cv_show('centrum', centrum)
                #cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\centrums\centrum.png', centrum)
            #else:

                #outer_contour_0 = cv2.imread('E:\jupyter\opencv\project\Extraction of PVP parameters\outer_contours\outer_contour_0.png')
                #inner_contour_1 = cv2.imread('E:\jupyter\opencv\project\Extraction of PVP parameters\inner_contours\inner_contour_0.png')
                #centrum = cv2.addWeighted(outer_contour_0, 1, inner_contour_1, 1, 0)
                #cv_show('centrum', centrum)
                #cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\centrums\centrum.png', centrum)


outer_contour_0 = cv2.imread('E:\jupyter\opencv\project\Extraction of PVP parameters\outer_contours\outer_contour_0.png')
inner_contour_0 = cv2.imread('E:\jupyter\opencv\project\Extraction of PVP parameters\inner_contours\inner_contour_0.png')
centrum = cv2.addWeighted(outer_contour_0, 1, inner_contour_0, 1, 0)
#cv_show('centrum', centrum)
cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\centrums\centrum.png', centrum)
#皮肤边界草图提取
#二值化
ret, skin_thresh = cv2.threshold(original_roi, 2, 255, cv2.THRESH_BINARY)
cv_show('thresh', skin_thresh)
kernel = np.ones((3, 3), np.uint8)
erosion = cv2.erode(skin_thresh, kernel)  # 腐蚀以避免皮肤与金属条连接
#轮廓提取
skin_binary, skin_contours, skin_hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
skin_draw_img = original_img.copy()
skin_pictures = cv2.drawContours(skin_draw_img, skin_contours, -1, (0, 0, 255), 1)
cv_show('pictures', skin_pictures)
# 计算轮廓周长
skin_perimeter_list = []
for skin_cnt in skin_contours:
    skin_perimeter = cv2.arcLength(skin_cnt, True)
    skin_perimeter_list.append(skin_perimeter)
skin_perimeter_list.sort(reverse=True)
#print(skin_perimeter_list)
#通过周长筛选出轮廓
skin_contours_len = []
for i in range(len(skin_contours)):
    skin_img_temp = np.zeros(original_img.shape, np.uint8)
    skin_contours_len.append(skin_img_temp)
    skin_length = cv2.arcLength(skin_contours[i], True)
    #设定保留轮廓的条件为轮廓周长最长的
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

#从左到右按列遍历像素点，保留每一列最后的像素点
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
centrums_0 = cv2.imread('E:\jupyter\opencv\project\Extraction of PVP parameters\centrums\centrum.png')
finall_skin_0 = cv2.imread('E:\jupyter\opencv\project\Extraction of PVP parameters\skin_contours\\finall_skin.png')
divide_result = cv2.addWeighted(centrums_0, 1, finall_skin_0, 1, 0)
#cv_show('divide_result', divide_result)
cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\divide_results\divide_result.png', divide_result)

#显示分割效果
divide_result_0 = cv2.imread('E:\jupyter\opencv\project\Extraction of PVP parameters\divide_results\divide_result.png')
#original_image_1 = cv2.imread('E:\jupyter\opencv\project\Extraction of PVP parameters\original_images\example_26.jpg')
merge = cv2.addWeighted(divide_result_0, 1, original_img, 0.7, 0)
cv_show('merge', merge)
cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\merge_images\merge.png', merge)


#参数提取
#图像旋转
#寻找椎体内轮廓的内接圆及圆心
inner_circle_gray = cv2.cvtColor(inner_contour_0, cv2.COLOR_BGR2GRAY)
# 二值化
ret, inner_circle_gray = cv2.threshold(inner_circle_gray, 10, 255, cv2.THRESH_BINARY)

# 识别轮廓
#contours, _ = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
inner_circle_binary, inner_circle_contours, inner_circle_hierarchy = cv2.findContours(inner_circle_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# 计算到轮廓的距离
inner_raw_dist = np.empty(inner_circle_gray.shape, dtype=np.float32)
for i in range(inner_circle_gray.shape[0]):
    for j in range(inner_circle_gray.shape[1]):
        inner_raw_dist[i, j] = cv2.pointPolygonTest(inner_circle_contours[0], (j, i), True)

# 获取最大值即内接圆半径，中心点坐标
inner_minVal, inner_maxVal, inner__, inner_maxDistPt = cv2.minMaxLoc(inner_raw_dist)
#print(inner_maxDistPt)
#average_2 = int(inner_maxDistPt[0])
#average_1 = int(inner_maxDistPt[1])
#inner_maxDistPt_2 = int(inner_maxDistPt[0])
#inner_maxDistPt_1 = int(inner_maxDistPt[1])

# 半径：maxVal  圆心：maxDistPt
# 转换格式
inner_maxVal = abs(inner_maxVal)
inner_radius = int(inner_maxVal)
# 原图转换格式
inner_result = cv2.cvtColor(inner_circle_gray, cv2.COLOR_GRAY2BGR)
# 绘制内切圆
cv2.circle(inner_result, inner_maxDistPt, inner_radius, (0, 255, 0), 1, 1, 0)
# 绘制圆心
cv2.circle(inner_result, inner_maxDistPt, 1, (0, 0, 255), 1, 1, 0)
#cv_show('0', inner_result)


para_inner_contour = cv2.imread('E:\jupyter\opencv\project\Extraction of PVP parameters\inner_contours\inner_contour_0.png')
#cv_show('para_inner_contour', para_inner_contour)
para_inner_ima = Image.open('E:\jupyter\opencv\project\Extraction of PVP parameters\inner_contours\inner_contour_0.png')
para_inner_im = pretreatment(para_inner_ima)
para_inner_index = []
for i in range(para_inner_im.shape[0]):
    for j in range(para_inner_im.shape[1]):
        if para_inner_im[i, j] == 1:
            para_inner_index.append([i, j])
"""
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
"""
#在椎体图像以及总体分割结果中把中心点画成红色
#centrums_1[average_1, average_2] = [0, 0, 255]
#divide_result_1[average_1, average_2] = [0, 0, 255]
#cv_show('divide_result_1', divide_result_1)
#cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\divide_results\divide_result_1.png', divide_result_1)
#cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\centrums\centrums_1.png', centrums_1)

"""
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
"""

"""#计算椎体外边界底角点
#提取椎体外边界坐标信息
outer_centrum_ima = Image.open('E:\jupyter\opencv\project\Extraction of PVP parameters\outer_contours\outer_contour_0.png')
outer_centrum_im = pretreatment(outer_centrum_ima)
para_outer_index = []
outer_end_point_list = []
Pointa = [average_1, average_2]
Pointb = [end_point[0], end_point[1]]
for i in range(outer_centrum_im.shape[0]):
    for j in range(outer_centrum_im.shape[1]):
        if outer_centrum_im[i, j] == 1:
            para_outer_index.append([i, j])
for z in range(len(para_outer_index)):
    distance = getDist_P2L(para_outer_index[z], Pointa, Pointb)
    if distance <= 1:
        outer_end_point_list.append(para_outer_index[z])
#print(outer_end_point_list)
end_point_outer = outer_end_point_list[-1]
"""
#寻找椎体外轮廓的内接圆及圆心
circle_gray = cv2.cvtColor(outer_contour_0, cv2.COLOR_BGR2GRAY)
# 二值化
ret, circle_gray = cv2.threshold(circle_gray, 10, 255, cv2.THRESH_BINARY)

# 识别轮廓
#contours, _ = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
circle_binary, circle_contours, circle_hierarchy = cv2.findContours(circle_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# 计算到轮廓的距离
raw_dist = np.empty(circle_gray.shape, dtype=np.float32)
for i in range(circle_gray.shape[0]):
    for j in range(circle_gray.shape[1]):
        raw_dist[i, j] = cv2.pointPolygonTest(circle_contours[0], (j, i), True)

# 获取最大值即内接圆半径，中心点坐标
minVal, maxVal, _, maxDistPt = cv2.minMaxLoc(raw_dist)
#print(maxDistPt)
# 半径：maxVal  圆心：maxDistPt
# 转换格式
maxVal = abs(maxVal)
radius = int(maxVal)
# 原图转换格式
result = cv2.cvtColor(circle_gray, cv2.COLOR_GRAY2BGR)
# 绘制内切圆
cv2.circle(result, maxDistPt, radius, (0, 255, 0), 1, 1, 0)
# 绘制圆心
cv2.circle(result, maxDistPt, 1, (0, 0, 255), 1, 1, 0)
#cv_show('0', result)
slope_0 = (inner_maxDistPt[1] - maxDistPt[1]) / (inner_maxDistPt[0] - maxDistPt[0] + 0.001)
ang = math.atan((slope_0 - 10000) / (1 + (10000 * slope_0)))*180/math.pi
#print(ang)
#旋转整个分割结果
rows, cols = divide_result_0.shape[:2]
M = cv2.getRotationMatrix2D((maxDistPt[0], maxDistPt[1]), ang, 1)
divide_result_1 = cv2.warpAffine(divide_result_0, M, (cols, rows))
#cv_show('divide_result_1', divide_result_1)
cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\divide_results\divide_result_2.png', divide_result_1)

#旋转椎体
rows, cols = centrums_0.shape[:2]
M = cv2.getRotationMatrix2D((maxDistPt[0], maxDistPt[1]), ang, 1)
centrums_1 = cv2.warpAffine(centrums_0, M, (cols, rows))
#cv_show('centrums_1', centrums_1)
cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\centrums\centrums_1.png', centrums_1)

#旋转内边界轮廓
rows, cols = inner_contour_0.shape[:2]
M = cv2.getRotationMatrix2D((maxDistPt[0], maxDistPt[1]), ang, 1)
inner_contour_0 = cv2.warpAffine(inner_contour_0, M, (cols, rows))
#cv_show('inner_contour_2', inner_contour_2)
cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\inner_contours\inner_contour_3.png', inner_contour_0)

#旋转外边界轮廓
rows, cols = outer_contour_0.shape[:2]
M = cv2.getRotationMatrix2D((maxDistPt[0], maxDistPt[1]), ang, 1)
outer_contour_1 = cv2.warpAffine(outer_contour_0, M, (cols, rows))
#cv_show('outer_contour_1', outer_contour_1)
cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\outer_contours\outer_contour_1.png', outer_contour_1)

#旋转皮肤边界

rows, cols = finall_skin_0.shape[:2]
M = cv2.getRotationMatrix2D((maxDistPt[0], maxDistPt[1]), ang, 1)
finall_skin_1 = cv2.warpAffine(finall_skin_0, M, (cols, rows))
#cv_show('finall_skin_1', finall_skin_1)
cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\skin_contours\\finall_skin_1.png', finall_skin_1)

#重新获取旋转后椎体内轮廓的内接圆及圆心
inner_circle_gray = cv2.cvtColor(inner_contour_0, cv2.COLOR_BGR2GRAY)
# 二值化
ret, inner_circle_gray = cv2.threshold(inner_circle_gray, 10, 255, cv2.THRESH_BINARY)

# 识别轮廓
#contours, _ = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
inner_circle_binary, inner_circle_contours, inner_circle_hierarchy = cv2.findContours(inner_circle_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# 计算到轮廓的距离
inner_raw_dist = np.empty(inner_circle_gray.shape, dtype=np.float32)
for i in range(inner_circle_gray.shape[0]):
    for j in range(inner_circle_gray.shape[1]):
        inner_raw_dist[i, j] = cv2.pointPolygonTest(inner_circle_contours[0], (j, i), True)

# 内接圆半径，中心点坐标
inner_minVal, inner_maxVal, inner__, inner_maxDistPt = cv2.minMaxLoc(inner_raw_dist)
#print(inner_maxDistPt)
average_2 = int(inner_maxDistPt[0])
average_1 = int(inner_maxDistPt[1])
#inner_maxDistPt[0] = int(inner_maxDistPt[0])
#inner_maxDistPt[1] = int(inner_maxDistPt[1])
# 半径：maxVal  圆心：maxDistPt
# 转换格式
inner_maxVal = abs(inner_maxVal)
inner_radius = int(inner_maxVal)
# 原图转换格式
inner_result = cv2.cvtColor(inner_circle_gray, cv2.COLOR_GRAY2BGR)
# 绘制内切圆
cv2.circle(inner_result, inner_maxDistPt, inner_radius, (0, 255, 0), 1, 1, 0)
# 绘制圆心
cv2.circle(inner_result, inner_maxDistPt, 1, (0, 0, 255), 1, 1, 0)
#cv_show('0', inner_result)

centrums_1[average_1, average_2] = [0, 0, 255]
divide_result_1[average_1, average_2] = [0, 0, 255]
#cv_show('divide_result_1', divide_result_1)
cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\divide_results\divide_result_1.png', divide_result_1)
cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\centrums\centrums_1.png', centrums_1)

#提取总体边界坐标信息
result_ima = Image.open('E:\jupyter\opencv\project\Extraction of PVP parameters\divide_results\divide_result_1.png')
result_im = pretreatment(result_ima)

#打印出像素为1的像素点的索引
top_index = []
for i in range(result_im.shape[0]):
    for j in range(result_im.shape[1]):
        if result_im[i, j] == 1:
            top_index.append([i, j])


#从左侧（R）进行参数提取
#通过关键点画线，获取直线AE表达式
spin_3 = divide_result_1.copy()
cv2.line(spin_3, (average_2, average_1), (maxDistPt[0], maxDistPt[1]), (0, 0, 255), 1)
slope_1 = ((-maxDistPt[1]) - (-average_1)) / (maxDistPt[0] - average_2 + 0.01)
b1 = -maxDistPt[1] - slope_1 * maxDistPt[0]
pre_end_point_1 = ((0 - b1) // slope_1 + 1 )
pre_end_point_2 = (-512 - b1) // slope_1
end_point_1 = int(pre_end_point_1)
end_point_2 = int(pre_end_point_2)
#画出结果
spin_4 = divide_result_1.copy()
cv2.line(spin_4, (end_point_1, 0), (end_point_2, 512), (0, 0, 255), 1)
#cv_show('spin4', spin_4)
# 寻找直线与皮肤边界的交点以及椎体前缘顶点
top_index_2 = []
Pointa = [0, end_point_1]
Pointb = [512, end_point_2]
for i in range(result_im.shape[0]):
    for j in range(result_im.shape[1]):
        if result_im[i, j] == 1:
            top_index_2.append([i, j])
skin_inter_points_list1 = []
for v in range(len(top_index_2)):
    distance = getDist_P2L(top_index_2[v], Pointa, Pointb)
    #print(distance)
    if distance <= 1:
        #print(distance)
        skin_inter_points_list1.append(top_index_2[v])
#print(skin_inter_points_list1)
top = skin_inter_points_list1[0]
skin_intersection_point_1 = skin_inter_points_list1[-1]
#print(top)
#print(skin_intersection_point_1)

#提取椎体边界坐标信息
centrums_ima = Image.open('E:\jupyter\opencv\project\Extraction of PVP parameters\centrums\centrums_1.png')
centrums_im = pretreatment(centrums_ima)

#通过椎体中心做一条起始线
fake_distance_list = []
intersection_point_1_list = []
intersection_point_2_list = []
count = -1
for k in range(30):
    spin_2 = divide_result_1.copy()
    Pointa = [average_1-8*k + 10, 50]
    Pointb = [average_1, average_2]
    cv2.line(spin_2, (Pointa[1], Pointa[0]), (Pointb[1], Pointb[0]), (0, 0, 255), 1)
    #cv_show('spin_2', spin_2)

    #找出交点，并计算两交点之间的距离，如果满足某条件，跳出循环
    # 遍历像素点计算距离
    # 打印出像素为1的像素点的索引
    top_index_1 = []
    for i in range(centrums_im.shape[1]):
        for j in range(centrums_im.shape[0]):
            if centrums_im[j, i] == 1:
                top_index_1.append([j, i])
    centrum_points = []
    for z in range(len(top_index_1)):
        distance = getDist_P2L(top_index_1[z], Pointa, Pointb)
        #print(distance)
        if distance <= 1:
            #print(distance)
            if top_index_1[z][1] <= average_2:
                centrum_points.append(top_index_1[z])
    #print(centrum_points)

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
                mid_point = [math.ceil(((intersection_point_1_list[count-1][0] + intersection_point_2_list[count-1][0])) / 2) , math.ceil(((intersection_point_1_list[count-1][1] + intersection_point_2_list[count-1][1])) / 2)]
                #print(intersection_point_1)
                #print(intersection_point_2)
                #print(k)
                #print(fake_distance_list[k-1])
                #print(mid_point)
                break

#画右边第一条线AC
cv2.line(spin_3, (top[1], top[0]), (mid_point[1], mid_point[0]), (0, 0, 255), 1)
slope_2 = ((-mid_point[0]) - (-top[0])) / (mid_point[1] - top[1])
b2 = -mid_point[0] - slope_2 * mid_point[1]
pre_end_point_3 = (-512 - b2) // slope_2
end_point_3 = int(pre_end_point_3)
cv2.line(spin_4, (top[1], top[0]), (end_point_3, 512), (0, 0, 255), 1)
#cv_show('spin_4', spin_4)

#寻找直线AC与椎体的交点F
centrum_ima = Image.open('E:\jupyter\opencv\project\Extraction of PVP parameters\centrums\centrums_1.png')
centrum_im = pretreatment(centrum_ima)
top_index_3 = []
Pointa = [top[0], top[1]]
Pointb = [512, end_point_3]
for i in range(centrum_im.shape[0]):
    for j in range(centrum_im.shape[1]):
        if centrum_im[i, j] == 1:
            top_index_3.append([i, j])
skin_inter_points_list2 = []
for z in range(len(top_index_3)):
    distance = getDist_P2L(top_index_3[z], Pointa, Pointb)
    #print(distance)
    if distance <= 1:
        #print(distance)
        skin_inter_points_list2.append(top_index_3[z])
#print(skin_inter_points_list2)
skin_intersection_point_2 = skin_inter_points_list2[-1]

#寻找直线与皮肤的交点C
skin_ima = Image.open('E:\jupyter\opencv\project\Extraction of PVP parameters\skin_contours\\finall_skin_1.png')
skin_im = pretreatment(skin_ima)
top_index_4 = []
Pointa = [top[0], top[1]]
Pointb = [512, end_point_3]
for i in range(skin_im.shape[0]):
    for j in range(skin_im.shape[1]):
        if skin_im[i, j] == 1:
            top_index_4.append([i, j])
skin_inter_points_list3 = []
for v in range(len(top_index_4)):
    distance = getDist_P2L(top_index_4[v], Pointa, Pointb)

    #print(distance)
    if distance <= 1:
        #print(distance)
        skin_inter_points_list3.append(top_index_4[v])
skin_inter_points_list3.sort()
#print(skin_inter_points_list3)
skin_intersection_point_3 = skin_inter_points_list3[0]

#通过交点C向椎体中心对称线做垂线CE，并导出交点E，用以获得旁开棘突距离CE
slope_3 = -1 / slope_1
#print(slope_3)
b3 = (-skin_intersection_point_3[0]) - (slope_3 * skin_intersection_point_3[1])
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

#通过交点F做与椎体中心线AE平行的线，寻找其与皮肤的交点G
b4 = -skin_intersection_point_2[0] - slope_1 * skin_intersection_point_2[1]
pre_end_point_4 = (-512 - b4) // slope_1
end_point_4 = int(pre_end_point_4)
#print(end_point_4)
cv2.line(spin_4, (skin_intersection_point_2[1], skin_intersection_point_2[0]), (end_point_4, 512), (0, 0, 255), 1)
#cv_show('spin4', spin_4)

top_index_5 = []
Pointa = [skin_intersection_point_3[0], skin_intersection_point_3[1]]
Pointb = [512, end_point_4]
for i in range(skin_im.shape[0]):
    for j in range(skin_im.shape[1]):
        if skin_im[i, j] == 1:
            top_index_5.append([i, j])
skin_inter_points_list4 = []
for v in range(len(top_index_5)):
    distance = getDist_P2L(top_index_5[v], Pointa, Pointb)
    #print(distance)
    if distance <= 1:
        #print(distance)
        skin_inter_points_list4.append(top_index_5[v])
#print(skin_inter_points_list4)
skin_intersection_point_4 = skin_inter_points_list4[-1]

#绘制结果
spin_5 = cv2.line(spin_4, (skin_intersection_point_3[1], skin_intersection_point_3[0]), (level_point[1], level_point[0]), (0, 0, 255), 1)
cv_show('spin_5', spin_5)
cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\spin_images\spin_5.png', spin_5)

#将结果旋转回原角度
rows, cols = spin_5.shape[:2]
M = cv2.getRotationMatrix2D((maxDistPt[0], maxDistPt[1]), -ang, 1)
spin_6 = cv2.warpAffine(spin_5, M, (cols, rows))
#cv_show('spin_6', spin_6)
cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\spin_images\spin_6.png', spin_6)

#将结果与原图融合
target = cv2.imread('E:\jupyter\opencv\project\Extraction of PVP parameters\spin_images\spin_6.png')
finall = cv2.addWeighted(target, 1, original_img, 0.7, 0)
cv_show('finall', finall)
cv2.imwrite('finall.png', finall)


"""
#从右侧（L）进行参数提取

#通过关键点画线，获取直线AE表达式
spin_3 = divide_result_1.copy()
cv2.line(spin_3, (average_2, average_1), (maxDistPt[0], maxDistPt[1]), (0, 0, 255), 1)
slope_1 = ((-maxDistPt[1]) - (-average_1)) / (maxDistPt[0] - average_2 + 0.01)
b1 = -maxDistPt[1] - slope_1 * maxDistPt[0]
pre_end_point_1 = ((0 - b1) // slope_1 + 1 )
pre_end_point_2 = (-512 - b1) // slope_1
end_point_1 = int(pre_end_point_1)
end_point_2 = int(pre_end_point_2)
#画出结果
spin_4 = divide_result_1.copy()
cv2.line(spin_4, (end_point_1, 0), (end_point_2, 512), (0, 0, 255), 1)
#cv_show('spin4', spin_4)
# 寻找直线与皮肤边界的交点以及椎体前缘顶点
top_index_2 = []
Pointa = [0, end_point_1]
Pointb = [512, end_point_2]
for i in range(result_im.shape[0]):
    for j in range(result_im.shape[1]):
        if result_im[i, j] == 1:
            top_index_2.append([i, j])
skin_inter_points_list1 = []
for v in range(len(top_index_2)):
    distance = getDist_P2L(top_index_2[v], Pointa, Pointb)
    #print(distance)
    if distance <= 1:
        #print(distance)
        skin_inter_points_list1.append(top_index_2[v])
#print(skin_inter_points_list1)
top = skin_inter_points_list1[0]
skin_intersection_point_1 = skin_inter_points_list1[-1]
#print(top)
#print(skin_intersection_point_1)

#提取椎体边界坐标信息
centrums_ima = Image.open('E:\jupyter\opencv\project\Extraction of PVP parameters\centrums\centrums_1.png')
centrums_im = pretreatment(centrums_ima)

#通过椎体中心做一条起始线
fake_distance_list = []
intersection_point_1_list = []
intersection_point_2_list = []
count = -1
for k in range(30):
    spin_2 = divide_result_1.copy()
    Pointa = [average_1-8*k + 20, 400]
    Pointb = [average_1, average_2]
    cv2.line(spin_2, (Pointa[1], Pointa[0]), (Pointb[1], Pointb[0]), (0, 0, 255), 1)
    #cv_show('spin_2', spin_2)

    #找出交点，并计算两交点之间的距离，如果满足某条件，跳出循环
    # 遍历像素点计算距离
    # 打印出像素为1的像素点的索引
    top_index_1 = []
    for i in range(centrums_im.shape[1]):
        for j in range(centrums_im.shape[0]):
            if centrums_im[j, i] == 1:
                top_index_1.append([j, i])
    centrum_points = []
    for z in range(len(top_index_1)):
        distance = getDist_P2L(top_index_1[z], Pointa, Pointb)
        #print(distance)
        if distance <= 1:
            #print(distance)
            if top_index_1[z][1] >= average_2:
                centrum_points.append(top_index_1[z])
    #print(centrum_points)

    if len(centrum_points) >= 3:
        count += 1
        intersection_point_1 = centrum_points[1]
        intersection_point_1_list.append(intersection_point_1)
        #print(intersection_point_1)
        intersection_point_2 = centrum_points[len(centrum_points)-1]
        intersection_point_2_list.append(intersection_point_2)
        #print(intersection_point_2)
        fake_distance = math.sqrt(((intersection_point_1[0] - intersection_point_2[0]) ** 2) + ((intersection_point_1[1] - intersection_point_2[1]) ** 2))
        #print(fake_distance)
        fake_distance_list.append(fake_distance)
        if k >= 1:
            if fake_distance_list[count] <= fake_distance_list[count-1]:
                continue
            else:
                mid_point = [math.ceil(((intersection_point_1_list[count-1][0] + intersection_point_2_list[count-1][0])) / 2) , math.ceil(((intersection_point_1_list[count-1][1] + intersection_point_2_list[count-1][1])) / 2)]
                #print(intersection_point_1)
                #print(intersection_point_2)
                #print(k)
                #print(fake_distance_list[k-1])
                #print(mid_point)
                break

#画右边第一条线AC
cv2.line(spin_3, (top[1], top[0]), (mid_point[1], mid_point[0]), (0, 0, 255), 1)
slope_2 = ((-mid_point[0]) - (-top[0])) / (mid_point[1] - top[1])
b2 = -mid_point[0] - slope_2 * mid_point[1]
pre_end_point_3 = (-512 - b2) // slope_2
end_point_3 = int(pre_end_point_3)
cv2.line(spin_4, (top[1], top[0]), (end_point_3, 512), (0, 0, 255), 1)
#cv_show('spin_4', spin_4)

#寻找直线AC与椎体的交点F
centrum_ima = Image.open('E:\jupyter\opencv\project\Extraction of PVP parameters\centrums\centrums_1.png')
centrum_im = pretreatment(centrum_ima)
top_index_3 = []
Pointa = [top[0], top[1]]
Pointb = [512, end_point_3]
for i in range(centrum_im.shape[0]):
    for j in range(centrum_im.shape[1]):
        if centrum_im[i, j] == 1:
            top_index_3.append([i, j])
skin_inter_points_list2 = []
for z in range(len(top_index_3)):
    distance = getDist_P2L(top_index_3[z], Pointa, Pointb)
    #print(distance)
    if distance <= 1:
        #print(distance)
        skin_inter_points_list2.append(top_index_3[z])
#print(skin_inter_points_list2)
skin_intersection_point_2 = skin_inter_points_list2[-1]

#寻找直线与皮肤的交点C
skin_ima = Image.open('E:\jupyter\opencv\project\Extraction of PVP parameters\skin_contours\\finall_skin_1.png')
skin_im = pretreatment(skin_ima)
top_index_4 = []
Pointa = [top[0], top[1]]
Pointb = [512, end_point_3]
for i in range(skin_im.shape[0]):
    for j in range(skin_im.shape[1]):
        if skin_im[i, j] == 1:
            top_index_4.append([i, j])
skin_inter_points_list3 = []
for v in range(len(top_index_4)):
    distance = getDist_P2L(top_index_4[v], Pointa, Pointb)

    #print(distance)
    if distance <= 1:
        #print(distance)
        skin_inter_points_list3.append(top_index_4[v])
skin_inter_points_list3.sort()
#print(skin_inter_points_list3)
skin_intersection_point_3 = skin_inter_points_list3[0]

#通过交点C向椎体中心对称线做垂线CE，并导出交点E，用以获得旁开棘突距离CE
slope_3 = -1 / slope_1
#print(slope_3)
b3 = (-skin_intersection_point_3[0]) - (slope_3 * skin_intersection_point_3[1])
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

#通过交点F做与椎体中心线AE平行的线，寻找其与皮肤的交点G
b4 = -skin_intersection_point_2[0] - slope_1 * skin_intersection_point_2[1]
pre_end_point_4 = (-512 - b4) // slope_1
end_point_4 = int(pre_end_point_4)
#print(end_point_4)
cv2.line(spin_4, (skin_intersection_point_2[1], skin_intersection_point_2[0]), (end_point_4, 512), (0, 0, 255), 1)
#cv_show('spin4', spin_4)

top_index_5 = []
Pointa = [skin_intersection_point_3[0], skin_intersection_point_3[1]]
Pointb = [512, end_point_4]
for i in range(skin_im.shape[0]):
    for j in range(skin_im.shape[1]):
        if skin_im[i, j] == 1:
            top_index_5.append([i, j])
skin_inter_points_list4 = []
for v in range(len(top_index_5)):
    distance = getDist_P2L(top_index_5[v], Pointa, Pointb)
    #print(distance)
    if distance <= 1:
        #print(distance)
        skin_inter_points_list4.append(top_index_5[v])
#print(skin_inter_points_list4)
skin_intersection_point_4 = skin_inter_points_list4[-1]

#绘制结果
spin_5 = cv2.line(spin_4, (skin_intersection_point_3[1], skin_intersection_point_3[0]), (level_point[1], level_point[0]), (0, 0, 255), 1)
#cv_show('spin_4', spin_4)
cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\spin_images\spin_5.png', spin_5)

#将结果旋转回原角度
rows, cols = spin_5.shape[:2]
M = cv2.getRotationMatrix2D((maxDistPt[0], maxDistPt[1]), -ang, 1)
spin_6 = cv2.warpAffine(spin_5, M, (cols, rows))
#cv_show('spin_6', spin_6)
cv2.imwrite('E:\jupyter\opencv\project\Extraction of PVP parameters\spin_images\spin_6.png', spin_6)

#将结果与原图融合
target = cv2.imread('E:\jupyter\opencv\project\Extraction of PVP parameters\spin_images\spin_6.png')
finall = cv2.addWeighted(target, 1, original_img, 0.7, 0)
cv_show('finall', finall)
cv2.imwrite('finall.png', finall)
"""





#参数计算过程
parameter_AC = math.sqrt((abs(top[1] - skin_intersection_point_3[1]) ** 2) + (abs(top[0] - skin_intersection_point_3[0]) ** 2))
parameter_AD = math.sqrt((abs(top[1] - skin_intersection_point_1[1]) ** 2) + (abs(top[0] - skin_intersection_point_1[0]) ** 2))
parameter_AE = math.sqrt((abs(top[1] - level_point[1]) ** 2) + (abs(top[0] - level_point[0]) ** 2))
parameter_CE = math.sqrt((abs(skin_intersection_point_3[1] - level_point[1]) ** 2) + (abs(skin_intersection_point_3[0] - level_point[0]) ** 2))
parameter_FG = math.sqrt((abs(skin_intersection_point_2[1] - skin_intersection_point_4[1]) ** 2) + (abs(skin_intersection_point_2[0] - skin_intersection_point_4[0]) ** 2))
parameter_CG = parameter_CE * (parameter_FG / parameter_AE)
parameter_GE = parameter_CE - parameter_CG
parameter_Angle = math.asin(parameter_CE / parameter_AC)*180/math.pi
#print('%.2f' % parameter_AC)
#print('%.2f' % parameter_AD)
#print('%.2f' % parameter_AE)
#print('%.2f' % parameter_CE)
#print('%.2f' % parameter_Angle)

#实际距离转化（具体转化比例还需进行标定）
rate = 0.3359375
AC = parameter_AC * rate
AD = parameter_AD * rate
AE = parameter_AE * rate
CE = parameter_CE * rate
FG = parameter_FG * rate
CG = parameter_CG * rate
GE = parameter_GE * rate
print("进针深度AC = ", '%.2f' % AC, "mm")
#print("AD = ", '%.2f' % AD, "mm")
#print("AE = ", '%.2f' % AE, "mm")
print("进针位置旁开棘突距离CE = ", '%.2f' % CE, "mm")
print("定位针初始距横突距离FG = ", '%.2f' % FG, "mm")
#print("CG = ", '%.2f' % CG, "mm")
print("G臂射线起始点旁开棘突距离GE = ", '%.2f' % GE, "mm")
print("入针角度Angle = ", '%.2f' % parameter_Angle, "°")
#图像尺寸大小与人体实际大小的转化