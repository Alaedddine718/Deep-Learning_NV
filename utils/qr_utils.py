import qrcode

def make_qr(data: str, out_path: str):
    img = qrcode.make(data)
    img.save(out_path)
    return out_path
