import cv2
import base64


def get_local_img_b64(img_file, img_type=None):
    with open(img_file, 'rb') as f:
        img_b64 = base64.b64encode(f.read())
        src_b64 = img_b64.decode()

    if not img_type:
        img_type = img_file.split('.')[-1]

    if img_type == 'ico':
        return "data:image/x-icon;base64,%s" % src_b64
    if img_type == 'gif':
        return "data:image/gif;base64,%s" % src_b64
    if img_type == 'png':
        return "data:image/png;base64,%s" % src_b64
    if img_type == 'jpg' or img_type == 'jpeg':
        return "data:image/jpeg;base64,%s" % src_b64
    return ""

def get_mat_img_b64(img_mat):
    img = cv2.imencode('.jpg', img_mat)[1]
    img_b64 = base64.b64encode(img)
    src_b64 = img_b64.decode()
    return "data:image/jpeg;base64,%s" % src_b64

def get_mat_img_b64_list(img_mat_list):
    img_b64_list = []
    for img_mat in img_mat_list:
        data = []
        for img in img_mat:
            if img is not None:
                data.append(get_mat_img_b64(img))
        img_b64_list.append(data)
    return img_b64_list

def get_img_mat_list():
    import os
    from app.view import IMG_DIR
    data = []
    data1 = []
    data2 = []
    data3 = []

    a = cv2.imread(os.path.join(IMG_DIR, "ql.jpg"))
    b = cv2.imread(os.path.join(IMG_DIR, "bh.jpg"))
    c = cv2.imread(os.path.join(IMG_DIR, "zq.jpg"))
    d = cv2.imread(os.path.join(IMG_DIR, "xu.jpg"))
    e = cv2.imread(os.path.join(IMG_DIR, "ws.jpg"))
    f = cv2.imread(os.path.join(IMG_DIR, "hp.jpg"))
    if a is not None:
        data1.append(a)
    if b is not None:
        data2.append(b)
    if c is not None:
        data2.append(c)
    if d is not None:
        data3.append(d)
    if e is not None:
        data3.append(e)
    if f is not None:
        data3.append(f)
    data.append(data1)
    data.append(data2)
    data.append(data3)
    return data