import argparse
from binascii import crc_hqx
from datetime import datetime
from os import listdir, makedirs, path, remove, rename, rmdir
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from cv2 import cv2 as cv
from pyexiv2 import Image, ImageData

# -- Global Variables --
dest_path = None
source_path = None
# -- Global Constants --
days = (0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)  # added dummy zero so that index = month
days_ly = (0, 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335)  # leap year days because that would be a nightmare to deal with 3 years from now
zip_magic = [0x50, 0x4B, 0x03, 0x04]
gif87_magic = [0x47, 0x49, 0x46, 0x38, 0x37, 0x61]
gif89_magic = [0x47, 0x49, 0x46, 0x38, 0x39, 0x61]
header_info = ((1, 'uint', 'header_type'), (2, 'uint', 'header_length'))
header_dict = {
    0: ('primary header', *header_info,
        (1, 'uint', 'file_type_code'), (4, 'uint', 'total_header_length'), (8, 'uint', 'data_field_length')),
    1: ('image structure', *header_info,
        (1, 'uint', 'bits_per_pixel'), (2, 'uint', 'columns'), (2, 'uint', 'rows'), (1, 'uint', 'compression_flag')),
    2: ('image navigation', *header_info,
        (32, 'chr', 'projection_name'), (4, 'int', 'column_scaling_factor'), (4, 'int', 'line_scaling_factor'),
        (4, 'int', 'column_offset'), (4, 'int', 'line_offset')),
    3: ('image data function', *header_info,
        (-1, 'chr', 'data_definition_block')),
    4: ('annotation', *header_info,
        (-1, 'chr', 'annotation_text')),
    5: ('time stamp', *header_info,
        (7, 'time', 'time_stamp')),
    6: ('ancillary text', *header_info,
        (-1, 'chr', 'ancillary_text')),
    7: ('key header', *header_info,
        (-1, 'chr', 'key_header_information')),
    128: ('sequencing Header', *header_info,
          (2, 'uint', 'image_id'), (2, 'uint', 'sequence'), (2, 'uint', 'start_column'),
          (2, 'uint', 'start_row'), (2, 'uint', 'max_segments'), (2, 'uint', 'max_columns'), (2, 'uint', 'max_rows')),
    129: ('product id Header', *header_info,
          (4, 'chr', 'signature'), (2, 'uint', 'product_id'), (2, 'uint', 'subproduct_id'), (2, 'uint', 'parameter'),
          (1, 'uint', 'compression_type')),
    130: ('header structure record', *header_info,
          (-1, 'chr', 'header_fields')),
    131: ('rice header', *header_info,
          (2, 'uint', 'flags_mask'), (1, 'uint', 'pixels_per_block'), (1, 'uint', 'unknown')),
    -1: ('unknown header', *header_info,
         (-1, 'chr', 'unknown_data'))
}


def parse_flags(flags: int) -> str:
    """

    :param flags:
    :return:
    """
    out = ''
    if flags & 0x01 != 0:
        out += 'A'
    if flags & 0x02 != 0:
        out += 'B'
    if flags & 0x04 != 0:
        out += 'I'
    if flags & 0x08 != 0:
        out += 'N'
    if flags & 0x10 != 0:
        out += 'T'
    if flags & 0x20 != 0:
        out += 'U'
    if flags & 0x40 != 0:
        out += 'W'
    if 'M' not in out:
        out += 'G'
    return out


def bcd_to_date(data):
    year = format(data[6], '02x')  # Last 2 Digits of Year
    day = format(data[5], 'x') + format(data[4] >> 4, 'x')  # Julian Day
    hour = format(data[4] & 0xF, 'x') + format(data[3] >> 4, 'x')
    minute = format(data[3] & 0xF, 'x') + format(data[2] >> 4, 'x')
    seconds = format(data[2] & 0xF, 'x') + format(data[1] >> 4, 'x')
    milsec = format(data[1] & 0xF, 'x') + format(data[0], '02x')  # Fractions of a second
    date = datetime.strptime(year + day, '%y%j').date()  # Convert to date object to parse Julian Day
    date_str = date.strftime('%Y-%m-%d')
    return f"{date_str} {hour}:{minute}:{seconds}.{milsec}"


def bcd_to_time_path(data) -> str:
    year = format(data[6], '02x')
    day = format(data[5], 'x') + format(data[4] >> 4, 'x')
    hour = format(data[4] & 0xF, 'x') + format(data[3] >> 4, 'x')
    # Actuarial tables say that I will not live to see this break.
    return f"20{year}\\{day}\\{hour}"


def get_time_path(timestamp: str, month=True) -> str:
    if month:
        year = timestamp[0:4]
        month = int(timestamp[4:6])
        day = int(timestamp[6:8])
        hour = timestamp[8:10]
        # I don't think the second half of this condition will ever be necessary.
        # hell, I doubt the first half of this condition will ever be necessary.
        if int(year) % 4 == 0 and (int(year) % 100 != 0 or int(year) % 400 == 0):
            return f"{year}\\{days_ly[month] + day}\\{hour}"
        else:
            return f"{year}\\{days[month] + day}\\{hour}"
    else:
        return f"{timestamp[0:4]}\\{timestamp[4:7]}\\{timestamp[7:9]}"


def check_magic(magic: list) -> str:
    if all([magic[i] == zip_magic[i] for i in range(4)]):
        return 'zip'
    if all([magic[i] == gif87_magic[i] for i in range(6)]):
        return 'gif87'
    if all([magic[i] == gif89_magic[i] for i in range(6)]):
        return 'gif89'
    return 'mundane'  # mundane files have no (identifiable) magic in them


def getHeaders(file: np.array) -> list[dict]:
    out = []
    ptr = 0
    length = 16
    while ptr < length:
        d = {}
        h_type = file[ptr]
        try:
            d['type'] = header_dict[h_type][0]
        except KeyError:
            h_type = -1
            d['type'] = header_dict[h_type][0]
        for field in header_dict[h_type][1:]:
            f_len = field[0]
            f_type = field[1]
            f_key = field[2]
            if f_type == 'uint':
                d[f_key] = 0
                for i in range(f_len):
                    d[f_key] += file[ptr + i] * (2 ** (8 * (f_len - i - 1)))
            elif f_type == 'int':
                d[f_key] = 0
                if file[ptr] > 0x7F:
                    d[f_key] += ((file[ptr] ^ 0xFF) + 1) * (2 ** (8 * (f_len - 1)))
                for i in range(1, f_len):
                    d[f_key] += file[ptr + i] * (2 ** (8 * (f_len - i - 1)))
            elif f_type == 'chr':
                if f_len < 0:
                    f_len = d['header_length'] - 3
                d[f_key] = "".join([chr(i) for i in file[ptr: ptr + f_len]])
            elif f_type == 'time':
                d[f_key] = 0
                for i in range(f_len):
                    d[f_key] += file[ptr + i] * (2 ** (8 * (f_len - i - 1)))
            else:
                raise ValueError(f"Unrecognized data type '{f_type}'. Must be type 'int', 'uint', 'chr', or 'time'")
            ptr += f_len
        if h_type == 0:
            length = d['total_header_length']
        out.append(d)
    return out


def writeImageFile(filename: str, file: np.array, headers: list[dict]) -> str:
    """
    Writes a file of LRIT filetype '0' to the Imagery directory

    :param filename: the name of the file to write; directory and extension are determined automatically
    :param file: numpy array of type uint8 representing the bytes of the file
    :param headers: list of LRIT headers for the file
    :return: string indicating success, skip, or specific error
    """
    structure, sequence = None, None
    for head in headers:
        h_type = head['header_type']
        if h_type == 1:
            structure = head
        if h_type == 128:
            sequence = head
    if structure is None:
        return f"bad input for file {filename}: Missing or broken Image Structure Header."
    if structure['bits_per_pixel'] != 8:
        return f"bad input for file {filename}: Invalid bits per pixel value"

    data_start = headers[0]['total_header_length']
    magic = check_magic(file[data_start:data_start + 6])
    if magic == 'gif87' or magic == 'gif89':
        if args:
            if not args.graphics:
                return "Skipped"
        timestamp = get_time_path(filename.split('-')[0])
        filename = f"{timestamp}\\{filename.split('-')[1]}"
        makedirs(f"{dest_path}\\Imagery\\{timestamp}", exist_ok=True)
        if path.isdir(f"{dest_path}\\Imagery\\{filename}.gif"):
            return "Skipped"
        with open(f"{dest_path}\\Imagery\\{filename}.gif", 'wb') as gif_file:
            gif_file.write(bytes(file[data_start:]))
        return "Success"
    if args:
        if not args.imagery:
            return "Skipped"
    seg = filename.split('-')
    timestamp = get_time_path(seg[3].split('_')[4][1:], month=False)

    # Make image directories if needed
    makedirs(f"{dest_path}\\Imagery\\{timestamp}", exist_ok=True)

    # sequenced vs. un-sequenced (mesoscale) Imagery
    if sequence is None:
        shape = (structure['rows'], structure['columns'])
        file_img = np.reshape(file[data_start:], shape)
        file_path = f"{dest_path}\\Imagery\\{timestamp}\\{seg[0]}-{seg[1]}-{seg[2]}-{seg[3].split('_')[0]}-{seg[3].split('_')[4][1:]}.png"
        if path.exists(file_path):
            return "Skipped"
        else:
            cv.imwrite(file_path, file_img)
            return "Success"
    else:
        # image is a multi-part file, get and assemble all LRIT files that make up this image
        structure = [structure]
        sequence = [sequence]
        file = [file]
        headers = [headers]
        file_path = f"{dest_path}\\Imagery\\{timestamp}\\{seg[0]}-{seg[1]}-{seg[2]}-{seg[3].split('_')[0]}-{sequence[0]['image_id']}.png"
        search_name = f"{'_'.join(filename.split('_')[:-1])}"
        chunks_towrite = [*range(sequence[0]['max_segments'])]

        # read image and metadata if it exists
        if path.exists(file_path):
            metadata = ImageData.read_comment(Image(file_path))
            chunks = [int(s) for s in metadata.split(',')]
            if sequence[0]['sequence'] in chunks:
                return f"Skipped"
            else:
                pass
            for c in chunks:
                if c in chunks_towrite:
                    chunks_towrite.remove(c)
            img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
        else:
            chunks = []
            img = np.zeros((sequence[0]['max_rows'], sequence[0]['max_columns']), dtype=np.uint8)
        chunks_found = []

        # find other LRIT files in sequence matching this image
        for c in chunks_towrite:
            name = f"{source_path}\\{search_name}_{str(c).rjust(3, '0')}.lrit"
            if name == f"{filename}.lrit":
                break
            if path.exists(name):
                new_file = np.fromfile(name, dtype=np.uint8)
                new_header = getHeaders(new_file)
                new_structure, new_sequence = None, None
                for head in new_header:
                    h_type = head['header_type']
                    if h_type == 1:
                        new_structure = head
                    if h_type == 128:
                        new_sequence = head
                if new_structure is None:
                    break
                if new_sequence is None:
                    break
                file.append(new_file)
                headers.append(new_header)
                structure.append(new_structure)
                sequence.append(new_sequence)
                chunks_found.append(new_sequence['sequence'])

        # use file data to assemble image
        for struct, seq, file_, head in zip(structure, sequence, file, headers):
            if seq['sequence'] not in chunks:
                data_start = head[0]['total_header_length']
                shape = (struct['rows'], struct['columns'])
                file_img = np.reshape(file_[data_start:], shape)
                chunks.append(seq['sequence'])
                start_row = seq['start_row']
                end_row = start_row + struct['rows']
                img[start_row:end_row][:] = file_img

        # finally write image and add metadata
        cv.imwrite(file_path, img)
        comment = ",".join([str(c) for c in chunks])
        ImageData.modify_comment(Image(file_path), comment)
    return f"Success"


def writeTextFile(filename: str, file: np.array, headers: list[dict]) -> str:
    """
    Writes a file of LRIT filetype '2' to the Text directory. Assumes files are uncompressed plaintext

    :param filename: the name of the file to write; directory and extension are determined automatically
    :param file: numpy array of type uint8 representing the bytes of the file
    :param headers: list of LRIT headers for the file
    :return: string indicating success, skip, or specific error
    """
    file_path = f"{dest_path}\\{filename}.txt"
    if filename[:23] != "GOES_EAST_Admin_message":
        parts = filename.split('_')
        timestamp = get_time_path(parts[4])
        filename = f"Text\\{timestamp}\\{filename}"
        file_path = f"{dest_path}\\{filename}.txt"
        if path.exists(file_path):
            return "Skipped"

        # make date directories if necessary
        makedirs(f"{dest_path}\\Text\\{timestamp}", exist_ok=True)
    elif path.exists(file_path):
        return "Skipped"

    data_start = headers[0]['total_header_length']
    with open(file_path, 'w', encoding='utf-8') as text_file:
        text_file.write(f"{filename}\n")
        text_file.write("==================== BEGIN HEADER ====================\n")
        for head in headers:
            text_file.write(f"----- {head['type']} -----\n")
            for k in head:
                if k != 'type':
                    text_file.write(f"\t{k}: {head[k]}\n")
        text_file.write("====================  END  HEADER ====================\n")
        text_file.write("".join([chr(i) for i in file[data_start:]]))
    return "Success"


def writeDCSFile(filename: str, file: np.array, headers: list[dict]) -> str:
    """
    Writes a file of LRIT filetype '130' to the DCS directory

    :param filename: the name of the file to write; directory and extension are determined automatically
    :param file: numpy array of type uint8 representing the bytes of the file
    :param headers: list of LRIT headers for the file
    :return: string indicating success, skip, or specific error
    """
    data_start = headers[0]['total_header_length']
    block_id = 0
    offset = 64 + data_start
    while block_id != 1:
        block_length = (file[offset + 2] << 8) | file[offset + 1]
        block_id = file[offset]
        if block_id != 1:
            offset += block_length
    timestamp = bcd_to_time_path(file[offset + 0x0C:offset + 0x13])
    filename = f"DCS\\{timestamp}\\{filename}.csv"
    if path.exists(f"{dest_path}\\{filename}"):
        return "Skipped"

    # make date directories if necessary
    makedirs(f"{dest_path}\\DCS\\{timestamp}", exist_ok=True)

    dcs_header = {'filename': ''.join([chr(c) for c in file[data_start + 0:data_start + 32]]),
                  'file_size': int(''.join([chr(c & 0x7F) for c in file[data_start + 32: data_start + 40]])),
                  'file_source': ''.join([chr(c) for c in file[data_start + 40:data_start + 44]]),
                  'file_type': ''.join([chr(c) for c in file[data_start + 44:data_start + 48]])}
    # I don't check the CRC because I'm a chad like that.
    offset = 64 + data_start
    with open(f"{dest_path}\\{filename}.csv", 'w', encoding='utf-8') as DCS_file:
        DCS_file.write(
            "size,seq_num,data_rate,platform,parity_error,ARM_flags,corrected_address,carrier_start,message_end,signal_strength,freq_offset,phs_noise,modulation_index,good_phs,channel,spacecraft,source_code,source_secondary,data,crc_ok\n"
        )
        while offset < dcs_header['file_size']:
            block_length = (file[offset + 2] << 8) | file[offset + 1]  # little endian ordering
            block = file[offset:offset + block_length]
            block_id = block[0]
            bauds = ['Undefined', '100', '300', '1200']
            scids = ['Unknown', 'GOES-East', 'GOES-West', 'GOES-Central', 'GOES-Test']  # Spacecraft ID's
            if block_id == 1:
                platforms = ['CS1', 'CS2']
                modulation_indicies = ['Unknown', 'Normal', 'High', 'Low']

                flags = block[6]
                abnormal_flags = block[7]
                output = [block_length,  # size
                          (block[5] << 16) | (block[4] << 8) | (block[3]),  # seq_num
                          bauds[flags & 0x07],  # data_rate
                          platforms[(flags & 0x08) >> 3],  # platform
                          (flags & 0x10) >> 4,  # parity_error
                          parse_flags(abnormal_flags),  # ARM_flags
                          (int(block[11]) << 24) | (block[10] << 16) | (block[9] << 8) | block[8],  # corrected_address
                          bcd_to_date(block[0x0C:0x13]),  # carrier_start
                          bcd_to_date(block[0x13:0x1A]),  # message_end
                          (((block[0x1B] << 8) | block[0x1A]) & 0x03FF) / 10,  # signal_strength
                          ((((block[0x1D] << 8) | block[0x1C]) & 0x03FF) - (
                              16384 if (((block[0x1D] << 8) | block[0x1C]) & 0x03FF > 8191) else 0)) / 10,
                          # freq_offset
                          (((block[0x1F] << 8) | block[0x1E]) & 0x01FF) / 100,  # phs_noise
                          modulation_indicies[(((block[0x1F] << 8) | block[0x1E]) & 0x0C000) >> 14],  # modulation_index
                          block[0x20] / 2,  # good_phs
                          ((block[0x22] << 8) | block[0x21]) & 0x03FF,  # channel
                          scids[((block[0x22] << 8) | block[0x21]) >> 12],  # spacecraft
                          "".join([chr(v & 0x7F) for v in block[0x23:0x25]]),  # source_code
                          "".join([chr(v & 0x7F) for v in block[0x25:0x27]]),  # secondary_source
                          "".join([f"{v & 0x7F}|" for v in block[0x27:-2]])  # data
                          ]
                dcp_crc16 = (block[-1] << 8) | block[-2]
                calc_crc = crc_hqx(block[:-2], 0xFFFF)

                if dcp_crc16 == calc_crc:
                    output.append(1)
                else:
                    output.append(0)
                [DCS_file.write(f"{key},") for key in output]
                DCS_file.write("\n")
            elif block_id == 2:
                # missed block
                pass
            offset += block_length
    return "Success"


def writeCompressedFile(filename: str, file: np.array, headers: list[dict]) -> str:
    """
    Decompresses and writes a file of compressed ZIP file to the appropriate directory(ies)

    :param filename: the name of the file to write; directory and extension are determined automatically
    :param file: numpy array of type uint8 representing the bytes of the file
    :param headers: list of LRIT headers for the file
    :return: string indicating success, skip, or specific error
    """
    file_dir = f"{dest_path}\\tmp\\{filename}"
    makedirs(file_dir, exist_ok=True)
    data_start = headers[0]['total_header_length']
    with open(f"{file_dir}.zip", 'wb') as zip_file:
        zip_file.write(bytes(file[data_start:]))
    with ZipFile(f"{file_dir}.zip") as zip_ref:
        zip_ref.extractall(file_dir)
    for unpacked in listdir(file_dir):
        unpacked_dir = f"{file_dir}\\{unpacked}"
        _, ext = unpacked.split('.')
        if args:
            if ext == 'TXT' and not args.text:
                break
            elif not args.graphics:
                break
        timestamp = get_time_path(unpacked.split('_')[4])
        output_dir = f"{dest_path}\\Text\\{timestamp}" if ext == 'TXT' else f"{dest_path}\\Imagery\\{timestamp}"
        makedirs(output_dir, exist_ok=True)
        if path.exists(f"{output_dir}\\{unpacked}"):
            remove(unpacked_dir)
            break
        rename(unpacked_dir, f"{output_dir}\\{unpacked}")
    remove(f"{file_dir}.zip")
    rmdir(file_dir)
    return "Success"


def writeData(filename: str, file: np.array, headers: list[dict]) -> str:
    """
    Writes an LRIT file to an appropriate directory, automatically determining filetype and compression.

    :param filename: the name of the file to write; directory and extension are determined automatically
    :param file: numpy array of type uint8 representing the bytes of the file
    :param headers: list of LRIT headers for the file
    :return: string indicating success, skip, or specific error
    """
    file_type = headers[0]['file_type_code']
    data_start = headers[0]['total_header_length']
    magic = check_magic(file[data_start:data_start + 6])
    if magic == 'zip':
        if args:
            if not (args.text or args.graphics):
                return "Skipped"
        return writeCompressedFile(filename, file, headers)
    if file_type == 0:
        if args:
            if not args.imagery or args.graphics:
                return "Skipped"
        # Image Data File (filename generated automatically, passed argument used for printing warnings)
        return writeImageFile(filename, file, headers)
    elif file_type == 2:
        if args:
            if not args.text:
                return "Skipped"
        # Alphanumeric Text File
        return writeTextFile(filename, file, headers)
    elif file_type == 130:
        if args:
            if not args.dcs:
                return "Skipped"
        # DCS File
        return writeDCSFile(filename, file, headers)
    else:
        return "Unrecognized file type"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='lritproc',
                                     usage='%(prog)s [options] in_path out_path',
                                     description='Process GOES LRIT files')
    parser.add_argument('in_path',
                        metavar='source',
                        type=str,
                        help='path to directory containing LRIT files', )
    parser.add_argument('out_path',
                        metavar='dest',
                        type=str,
                        help='path to directory to put output files', )
    output_group = parser.add_argument_group('output arguments')
    output_group.add_argument('-t', '--text',
                              help='process files containing KWIN plaintext',
                              action='store_true',
                              required=False)
    output_group.add_argument('-g', '--graphics',
                              help='process files containing KWIN graphics',
                              action='store_true',
                              required=False)
    output_group.add_argument('-i', '--imagery',
                              help='process files containing ABI imagery',
                              action='store_true',
                              required=False)
    output_group.add_argument('-d', '--dcs',
                              help='process files containing DCS blocks',
                              action='store_true',
                              required=False)
    output_group.add_argument('-a', '--all',
                              help='process all LRIT files',
                              action='store_true',
                              required=False)
    parser.add_argument('-v', '--verbose',
                        help='verbose mode',
                        action='store_true',
                        required=False)
    parser.add_argument('-q', '--quiet',
                        help='silences all prints',
                        action='store_true',
                        required=False)
    parser.add_argument('--mkdir',
                        help="make output directory if it doesn't exist (not recommended). lazy bastard",
                        action='store_true',
                        required=False)
    args = parser.parse_args()
    if not path.isdir(args.in_path):
        print(f"Source path '{args.in_path}' does not exist")
        exit(-1)
    if not path.isdir(args.out_path):
        if not args.mkdir:
            print(f"Destination path '{args.out_path}' does not exist")
            exit(-1)
        else:
            makedirs(args.out_path)
    if args.all:
        args.text = True
        args.imagery = True
        args.graphics = True
        args.dcs = True
    if not (args.text | args.imagery | args.graphics | args.dcs):
        print(f"No output filetypes specified")
        exit(-1)
    source_path = str(args.in_path)
    dest_path = str(args.out_path)
    start = perf_counter()
    for lrit_file in listdir(source_path):
        f_start = perf_counter()
        f = np.fromfile(f"{source_path}\\{lrit_file}", np.uint8)
        h = getHeaders(f)
        result = writeData(lrit_file.split('.')[0], f, h)
        f_end = perf_counter()
        if ("Success", "Skipped").__contains__(result) and args.verbose:
            print(f"processed {lrit_file} in {(f_end - f_start) * 1000:4.1f}ms - {result}")
        elif not args.quiet:
            print(f"Error for file {lrit_file}: {result}")
    end = perf_counter()
    if not args.quiet:
        print(f"all LRIT files successfully processed in {end - start:.4f} seconds")
    if path.isdir(f"{dest_path}\\tmp"):
        rmdir(f"{dest_path}\\tmp")
