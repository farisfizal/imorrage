import numpy as np
import cv2
from matplotlib import pyplot as plt

def rmv_bad_contours(source,contours):

    max_cnt_area = 0
    bad_cnts = []
    rows, cols = img.shape
    bad_mask = np.zeros((rows, cols, 1), np.uint8)

    for i in range(np.size(contours) - 1):
        if cv2.contourArea(contours[i]) < cv2.contourArea(contours[i+1]):
            max_cnt_area = i + 1

    for i in range(np.size(contours)):
        if cv2.contourArea(contours[i]) < 1:
            bad_cnts.append(i)
        else:
            bad_M = cv2.moments(contours[i])
            bad_cx = int(bad_M['m10']/bad_M['m00'])
            bad_cy = int(bad_M['m01']/bad_M['m00'])
            if cv2.pointPolygonTest(contours[max_cnt_area],(bad_cx,bad_cy),False) < 0:
                bad_cnts.append(i)

    bad_mask_inv = cv2.bitwise_not(bad_mask)
    clean = cv2.bitwise_and(source,source,mask=bad_mask_inv)

    return clean

def otsu_threshold(img):

    blur = cv2.GaussianBlur(img,(41,41),0)
    _,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return thresh

def find_edges(img):

    edges = cv2.Canny(img,100,200,L2gradient=True)

    return edges

def find_contours(img):

    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours, hierarchy

def contour_approx(contours):

    approx_contours = []
    for i in range(np.size(contours)):
        epsilon = 0.0005*cv2.arcLength(contours[i],True)
        approx = cv2.approxPolyDP(contours[i],epsilon,True)
        approx_contours.append(approx)

    return approx_contours

def draw_contours(source, approx_contours, boundRect, centers, radii, input_width):

    lat_R = False
    lat_L = False
    presence = False
    scan_center = None

    for i in range(np.size(approx_contours)):
        if (boundRect[i][2] * boundRect[i][3]) < 50 or (boundRect[i][2] * boundRect[i][3]) > 300000:
            M_scan = cv2.moments(approx_contours[i])
            scan_center = int(M_scan['m10']/M_scan['m00'])
            break
        elif boundRect[i][3] > 425:
            M_scan = cv2.moments(approx_contours[i])
            scan_center = int(M_scan['m10']/M_scan['m00'])
            break

    for i in range(np.size(approx_contours)):
        if (boundRect[i][2] * boundRect[i][3]) < 50 or (boundRect[i][2] * boundRect[i][3]) > 300000:
            cv2.drawContours(source, approx_contours, i, (255, 255, 255), 4)
        elif boundRect[i][3] > 425:
            cv2.drawContours(source, approx_contours, i, (255, 255, 255), 4)
        elif scan_center != None:
            presence = True
            M = cv2.moments(approx_contours[i])
            cx = int(M['m10']/M['m00'])
            if cx - 50 >= scan_center:
                lat_R = True
            elif cx + 50 <= scan_center:
                lat_L = True
            cv2.drawContours(source, approx_contours, i, (0, 0, 0), 5)
            cv2.drawContours(source, approx_contours, i, (255, 35, 35), 5)

    source_rgba = cv2.cvtColor(source,cv2.COLOR_RGB2RGBA)

    return source_rgba, presence, lat_R, lat_L

def find_child_contours(contours, hierarchy):

    child_contours = []

    for i in range(np.size(contours)):
        if hierarchy[0][i][2] == -1 and i != np.size(contours) - 1 and hierarchy[0][i+1][2] == -1:
            child_contours.append(contours[i])

    return child_contours

def poly_contour_approx(contours):

    boundRect = [None]*len(contours)
    centers = [None]*len(contours)
    radii = [None]*len(contours)
    for i, c in enumerate(contours):
        boundRect[i] = cv2.boundingRect(contours[i])
        centers[i], radii[i] = cv2.minEnclosingCircle(contours[i])

    return boundRect, centers, radii

def hemor_detect(file_name):

    # Read in the image file and vascular regions image file

    img = cv2.imread(file_name,0)
    vascular_regions = cv2.imread('vascular_labelled.png', -1)

    # Process vascular regions image file for addition with output image

    vascular_img_height, vascular_img_width, vascular_img_channels = vascular_regions.shape

    img_colour = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    img_rgba = cv2.cvtColor(img_colour, cv2.COLOR_RGB2RGBA)

        # Scaling images for overlaying vascular regions labelling onto output image
            # First, scaling height of vascular labelling

    input_height, input_width, input_channels = img_rgba.shape

    if vascular_img_height > input_height:
        vascular_scale = (input_height / vascular_img_height) * 0.925
    elif vascular_img_height < input_height:
        vascular_scale = (vascular_img_height / input_height) * 0.925

    scaled_width = int(vascular_regions.shape[1] * vascular_scale)
    scaled_height = int(vascular_regions.shape[0] * vascular_scale)
    dim = (scaled_width, scaled_height)

    vascular_regions_scaled = cv2.resize(vascular_regions, dim, interpolation=cv2.INTER_AREA)

    vascular_scaled_height, vascular_scaled_width, vascular_scaled_channels = vascular_regions_scaled.shape

            # Next, scaling width of vascular labelling

    if vascular_scaled_height > input_height:
        vascular_scale_two = (input_height / vascular_scaled_height) * 0.925
    elif vascular_scaled_height < input_height:
        vascular_scale_two = (vascular_scaled_height / input_height) * 0.925

    scaled_width_two = int(vascular_regions_scaled.shape[1] * vascular_scale_two)
    scaled_height_two = int(vascular_regions_scaled.shape[0] * vascular_scale_two)
    dim_two = (scaled_width_two, scaled_height_two)

    vascular_regions_scaled_two = cv2.resize(
        vascular_regions_scaled, dim_two, interpolation=cv2.INTER_AREA)

    vascular_scaled_height_two, vascular_scaled_width_two, vascular_scaled_channels_two = vascular_regions_scaled_two.shape

    top_left_x = round((input_width - scaled_width_two) / 2)
    top_left_y = round((input_height - scaled_height_two) / 2)

        # Creating vascular labelling overlay

    bkg_vascular_img = np.zeros(img_rgba.shape, np.uint8)

    bkg_vascular_img[top_left_y:(top_left_y + vascular_scaled_height_two), top_left_x:(
        top_left_x + vascular_scaled_width_two)] = vascular_regions_scaled_two

    # Apply Gaussian filtering, then Otsu's thresholding

    thresh = otsu_threshold(img)

    # Find edges

    edges_source = find_edges(thresh)
    edges_contour = edges_source

    # Find contours

    contours, hierarchy = find_contours(edges_contour)

    # Draw skull contours in black, hemorrhage contours in red

    approx_contours = contour_approx(contours)

    boundRect, centers, radii = poly_contour_approx(approx_contours)

    bkg = cv2.cvtColor(np.zeros(edges_contour.shape, np.uint8), cv2.COLOR_GRAY2RGB)
    bkg_vascular_img = cv2.cvtColor(bkg_vascular_img, cv2.COLOR_RGBA2RGB)

    final_img_rgba, presence, _, _ = draw_contours(bkg, approx_contours, boundRect, centers, radii, input_width)
    final_img_vascular, presence, lat_R, lat_L = draw_contours(bkg_vascular_img, approx_contours, boundRect, centers, radii, input_width)

    # Output contours image, contours image with labelled vascular regions

    filename_split = file_name.split('.')

    description = open(f'{filename_split[0]}_analysis_out.txt', 'w+')

    if presence == True:
        description.write("There is hemorrhaging present in this cranial CT scan.\n")
        txt = "There is hemorrhaging present in this cranial CT scan. "

        if lat_R == True:
            description.write(
                "The hemorrhaging is primarily in the right hemisphere of the patient's brain.")
            txt += "The hemorrhaging is primarily in the right hemisphere of the patient's brain."
        elif lat_L == True:
            description.write(
                "The hemorrhaging is primarily in the left hemisphere of the patient's brain.")
            txt += "The hemorrhaging is primarily in the left hemisphere of the patient's brain."
        elif lat_R == True and lat_L == True:
            description.write(
                "The hemorrhaging is present in both hemispheres of the patient's brain.")
            txt += "The hemorrhaging is present in both hemispheres of the patient's brain."
        else:
            description.write(
                "The hemorrhaging is not localized to one or both hemispheres.")
            txt += "The hemorrhaging is not localized to one or both hemispheres."

        # Two output images: one with outlined hemorrhaging and one with a vascular region overlay

        #plt.figure(num='Outlined Hemorrhage Sites with Labelled Vascular Regions')
        #plt.imshow(final_img_vascular, cmap='Greys', interpolation='bicubic')
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.savefig('out/vascular_out.png')

        #plt.figure(num='Outlined Hemorrhage Sites without Labelled Vascular Regions')
        #plt.imshow(final_img_rgba, cmap='Greys', interpolation='bicubic')
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.savefig('out/out.png')

        txt += "\n"

        plt.subplot(1, 2, 1), plt.imshow(final_img_rgba, 'gray')
        plt.title('Hemorrhage Sites'), plt.xticks([]), plt.yticks([])
        plt.subplot(1, 2, 2), plt.imshow(final_img_vascular, 'gray')
        plt.title('Vascular Regions Colormap'), plt.xticks([]), plt.yticks([])
        plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
        plt.savefig(f'{filename_split[0]}_out.png')

        description.close()
    else:
        description.write(
            "There is no hemorrhaging present in this cranial CT scan.")
        txt = "There is no hemorrhaging present in this cranial CT scan.\n"
        plt.title('No Hemorrhaging Detected')
        plt.imshow(cv2.bitwise_not(edges_contour), cmap='Greys', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
        plt.savefig(f'{filename_split[0]}_none.png')

        description.close()

print("Enter the file name: ")
name_input = input()

hemor_detect(name_input)
