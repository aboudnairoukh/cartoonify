##################################################################
# FILE : cartoonify.py
# WRITER : Abed EL Rahman Nairoukh , aboudnairoukh , 213668700
# EXERCISE : intro2cs2 ex6 2022
# DESCRIPTION : a program that cartoonify an image
##################################################################
import sys
import math
import ex6_helper


def input_checking():
    """This function checks the parameter's validity"""
    if len(sys.argv) != 8:
        print("There is a problem with the parameter's number")
        return None
    else:
        image_source = sys.argv[1]
        cartoon_dest = sys.argv[2]
        max_im_size = int(sys.argv[3])
        blur_size = int(sys.argv[4])
        th_block_size = int(sys.argv[5])
        th_c = int(sys.argv[6])
        quant_num_shades = int(sys.argv[7])
        return (image_source, cartoon_dest, max_im_size, blur_size,
                th_block_size, th_c, quant_num_shades)


def separate_channels(image):
    """This function separates the channels of the given image"""
    seperated_channels = []
    for z in range(len(image[0][0])):
        row_list = []
        for i in range(len(image)):
            column_list = []
            for column in image[i]:
                column_list.append(column[z])
            row_list.append(column_list)
        seperated_channels.append(row_list)
    return seperated_channels


def combine_channels(channels):
    """This function combines the channels of the given image"""
    combined_channels = []
    for i in range(len(channels[0])):
        columns_list = []
        for j in range(len(channels[0][0])):
            channels_list = []
            for z in range(len(channels)):
                channels_list.append(channels[z][i][j])
            columns_list.append(channels_list)
        combined_channels.append(columns_list)
    return combined_channels


def RGB2grayscale(colored_image):
    """This function transform a coloured image to a grayscale image"""
    greyscale_image = []
    for i in range(len(colored_image)):
        greyscale_image_row = []
        for j in range(len(colored_image[i])):
            greyscale_image_row.append(
                round(colored_image[i][j][0] * 0.299 + colored_image[i][j][1] * 0.587 +
                      colored_image[i][j][2] * 0.114))
        greyscale_image.append(greyscale_image_row)
    return greyscale_image


def blur_kernel(size):
    """This function returns a kernel with the size of size x size"""
    return [[1 / size ** 2] * size] * size


def sum_of_surrounding(y, x, image, limit_of_surrounding, kernel):
    """This function calculates the sum of the surrounding of the pixel"""
    size = len(kernel)
    pixel_sum = 0
    y_index = y - limit_of_surrounding
    for i in range(size):
        x_index = x - limit_of_surrounding
        for j in range(size):
            if 0 <= x_index < len(image[0]) and 0 <= y_index < len(image):
                num = image[y_index][x_index]
            else:
                num = image[y][x]
            pixel_sum += (kernel[i][j] * num)
            x_index += 1
        y_index += 1
    return pixel_sum


def calculate_kernel_pixel(y, x, image, kernel):
    """This function calculates a pixel applied by a kernel"""
    pixel_sum = round(sum_of_surrounding(y, x, image, int(len(kernel) / 2),
                                         kernel))
    if pixel_sum < 0:
        pixel_sum = 0
    elif pixel_sum > 255:
        pixel_sum = 255
    return pixel_sum


def apply_kernel(image, kernel):
    """This function applies the given kernel on the given image"""
    single_channel_image = []
    for y in range(len(image)):
        single_channel_image_row = []
        for x in range(len(image[y])):
            single_channel_image_row.append(
                calculate_kernel_pixel(y, x, image, kernel))
        single_channel_image.append(single_channel_image_row)
    return single_channel_image


def bilinear_interpolation(image, y, x):
    """This function calculates the pixel's amount in the destination image"""
    if y == len(image) - 1:
        y_rounded_up = int(y)
        y_rounded_down = y_rounded_up - 1
    else:
        y_rounded_down = int(y)
        y_rounded_up = y_rounded_down + 1
    if x == len(image[0]) - 1:
        x_rounded_up = int(x)
        x_rounded_down = x_rounded_up - 1
    else:
        x_rounded_down = int(x)
        x_rounded_up = x_rounded_down + 1
    delta_y = abs(y_rounded_down - y)
    delta_x = abs(x_rounded_down - x)
    return round(
        image[y_rounded_down][x_rounded_down] * (1 - delta_y) * (1 - delta_x)
        + image[y_rounded_up][x_rounded_down] * delta_y * (1 - delta_x)
        + image[y_rounded_down][x_rounded_up] * delta_x * (1 - delta_y)
        + image[y_rounded_up][x_rounded_up] * delta_x * delta_y)


def source_index(y, x, image, new_height, new_width):
    """This function returns the indexes y, x in the source image"""
    return (y / (new_height - 1) * (len(image) - 1)), \
           (x / (new_width - 1) * (len(image[0]) - 1))


def resize(image, new_height, new_width):
    """This function returns the given image in the size of
    new_height x new_width"""
    new_size_image = []
    for i in range(new_height):
        new_row = []
        for j in range(new_width):
            if i == 0 and j == 0:
                new_row.append(image[0][0])
            elif i == 0 and j == (new_width - 1):
                new_row.append(image[0][len(image[0]) - 1])
            elif i == (new_height - 1) and j == 0:
                new_row.append(image[len(image) - 1][0])
            elif i == (new_height - 1) and j == (new_width - 1):
                new_row.append(image[len(image) - 1][len(image[0]) - 1])
            else:
                y, x = source_index(i, j, image, new_height, new_width)
                new_row.append(bilinear_interpolation(image, y, x))
        new_size_image.append(new_row)
    return new_size_image


def scale_down_colored_image(image, max_size):
    """This function check if the image is smaller than max_size, if yes it
    returns None, else it returns a new image with the size if max_size"""
    height = len(image)
    width = len(image[0])
    if height <= max_size and width <= max_size:
        return None
    else:
        new_image_channels = []
        if height < width:
            new_height = round(height * (max_size / width))
            for channel in separate_channels(image):
                new_image_channels.append(
                    resize(channel, new_height, max_size))
        else:
            new_width = round(width * (max_size / height))
            for channel in separate_channels(image):
                new_image_channels.append(resize(channel, max_size, new_width))
        return combine_channels(new_image_channels)


def rotate_90(image, direction):
    """This function rotate the image 90 degree to the given side"""
    new_image = []
    if direction == 'L':
        for j in range(len(image[0]) - 1, -1, -1):
            new_row = []
            for i in range(len(image)):
                new_row.append(image[i][j])
            new_image.append(new_row)
    else:
        for j in range(len(image[0])):
            new_row = []
            for i in range(len(image) - 1, -1, -1):
                new_row.append(image[i][j])
            new_image.append(new_row)
    return new_image


def get_edges(image, blur_size, block_size, c):
    """This function returns the given image as an image of edges only"""
    blurred_image = apply_kernel(image, blur_kernel(blur_size))
    new_image = []
    r = block_size // 2
    for i in range(len(blurred_image)):
        new_image_row = []
        for j in range(len(blurred_image[i])):
            surrounding_avg = sum_of_surrounding(i, j, blurred_image, r,
                                                 blur_kernel(block_size))
            threshold = surrounding_avg - c
            if blurred_image[i][j] < threshold:
                new_image_row.append(0)
            else:
                new_image_row.append(255)
        new_image.append(new_image_row)
    return new_image


def quantize(image, N):
    """This function returns the given image as an image with fewer colors"""
    new_image = []
    for i in range(len(image)):
        new_image_row = []
        for j in range(len(image[i])):
            new_image_row.append(round(math.floor(image[i][j] * N / 256) *
                                    255 / (N - 1)))
        new_image.append(new_image_row)
    return new_image


def quantize_colored_image(image, N):
    """This function returns the given colorful image as an image
    with fewer colors"""
    new_image_channels = []
    for channels in separate_channels(image):
        new_image_channels.append(quantize(channels, N))
    return combine_channels(new_image_channels)


def add_mask(image1, image2, mask):
    """This function returns an image that is masked by image1 and image2"""
    new_image_channels = []
    three_dimension = type(image1[0][0]) == list
    if three_dimension:
        channels1 = separate_channels(image1)
        channels2 = separate_channels(image2)
    else:
        channels1 = [image1]
        channels2 = [image2]
    for c in range(len(channels1)):
        new_image_rows = []
        for i in range(len(channels1[c])):
            new_image_columns = []
            for j in range(len(channels1[c][i])):
                new_image_columns.append(round(channels1[c][i][j] * mask[i][j]
                                               + channels2[c][i][j] *
                                               (1 - mask[i][j])))
            new_image_rows.append(new_image_columns)
        new_image_channels.append(new_image_rows)
    if three_dimension:
        return combine_channels(new_image_channels)
    else:
        return new_image_channels[0]


def greyscale_to_mask(greyscale_image):
    """This function turns the given greyscale image to a mask"""
    mask = []
    for i in range(len(greyscale_image)):
        mask_row = []
        for j in range(len(greyscale_image[i])):
            mask_row.append(greyscale_image[i][j]/255)
        mask.append(mask_row)
    return mask


def cartoonify(image, blur_size, th_block_size, th_c, quant_num_shades):
    """This function returns the given image as a cartoonified image"""
    greyscale_image = RGB2grayscale(image)
    edges_image = get_edges(greyscale_image, blur_size, th_block_size,
                            th_c)
    quantized_image = quantize_colored_image(image, quant_num_shades)

    mask = greyscale_to_mask(edges_image)
    masked_image_channels = []
    for channel in separate_channels(quantized_image):
        masked_image_channels.append(add_mask(channel, edges_image, mask))
    return combine_channels(masked_image_channels)


if __name__ == '__main__':
    # parameters = input_checking()
    # if parameters is not None:
    #     (image_source, cartoon_dest, max_im_size, blur_size, th_block_size,
    #      th_c, quant_num_shades) = parameters
    #     image = ex6_helper.load_image(image_source)
    #     small_image = scale_down_colored_image(image, max_im_size)
    #     if small_image is not None:
    #         cartoonified_image = cartoonify(small_image, blur_size,
    #                                         th_block_size, th_c,
    #                                         quant_num_shades)
    #     else:
    #         cartoonified_image = cartoonify(image, blur_size, th_block_size,
    #                                         th_c, quant_num_shades)
    #     ex6_helper.save_image(cartoonified_image, cartoon_dest)
    print(blur_kernel(3))