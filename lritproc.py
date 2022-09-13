import argparse
import copy
from json import JSONEncoder, loads
from binascii import crc_hqx
from datetime import datetime
from os import listdir, makedirs, path, remove, rename, rmdir, scandir
from time import perf_counter
from typing import Optional
from zipfile import ZipFile

import numpy as np
from cv2 import cv2 as cv
from pyexiv2 import Image, ImageData

import logger

# -- Global Variables --
dest_path = None
source_path = None
manifest_updates = []
# -- Global Constants --
days = (0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304,
        334)  # added dummy zero so that index starts with 1; https://xkcd.com/163/
days_ly = (0, 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305,
           335)  # leap year days because that would be a nightmare to deal with 3 years from now
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
with open(r'dictionaries.json', 'r') as dicts:
    lookup = loads(''.join(dicts.readlines()))

logger = logger.logger(do_print=True)


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
    return f'{date_str} {hour}:{minute}:{seconds}.{milsec}'


def bcd_to_time_path(data) -> str:
    year = format(data[6], '02x')
    day = format(data[5], 'x') + format(data[4] >> 4, 'x')
    hour = format(data[4] & 0xF, 'x') + format(data[3] >> 4, 'x')
    # Actuarial tables say that I will not live to see this break.
    return f'20{year}./{day}/{hour}'


def get_time_path(timestamp: str, month=True) -> str:
    if month:
        year = timestamp[0:4]
        month = int(timestamp[4:6])
        day = int(timestamp[6:8])
        hour = timestamp[8:10]
        # I don't think the second half of this condition will ever be necessary.
        # hell, I doubt the first half of this condition will ever be necessary.
        if int(year) % 4 == 0 and (int(year) % 100 != 0 or int(year) % 400 == 0):
            return f'{year}/{days_ly[month] + day}/{hour}'
        else:
            return f'{year}/{days[month] + day}/{hour}'
    else:
        return f'{timestamp[0:4]}/{timestamp[4:7]}/{timestamp[7:9]}'


def check_magic(magic: list) -> str:
    if len(magic) < 6:
        return 'what the hell?'
    if all([magic[i] == zip_magic[i] for i in range(4)]):
        return 'zip'
    if all([magic[i] == gif87_magic[i] for i in range(6)]):
        return 'gif87'
    if all([magic[i] == gif89_magic[i] for i in range(6)]):
        return 'gif89'
    return 'mundane'  # mundane files have no (identifiable) magic in them\


def signed(val: int, bits: int) -> int:
    max_val = 2 ** (bits - 1)
    return (val % max_val) - (max_val * ((val // max_val) % 2))


def get_header_record(file: np.array) -> list[tuple]:
    record = []
    ptr = 0
    length = 16
    while ptr < length:
        h_type = file[ptr]
        h_len = (file[ptr + 1] * 256) | file[ptr + 2]
        if h_type == 0:
            length = (file[ptr + 5] * (256 ** 3)) | (file[ptr + 5] * (256 ** 2)) | (file[ptr + 6] * (256 ** 1)) | file[ptr + 7]
        record.append((h_type, ptr, h_len))
        ptr += h_len
    return record


def get_primary(file: np.array) -> dict:
    return {
        'type': 'primary',
        'header_type': 0,
        'header_length': 16,
        'file_type_code': file[3],
        'total_header_length': (file[4] * (256 ** 3)) + (file[5] * (256 ** 2)) + (file[6] * (256 ** 1)) + (file[7] * (256 ** 0)),
        'data_field_length': sum([file[8 + k] * (256 ** (7 - k)) for k in range(8)])
    }


def get_image_structure(file: np.array, record: list[tuple]) -> Optional[dict]:
    has_correct_header = False
    h_index = None
    for h, i, _ in record:
        if h == 1:
            has_correct_header = True
            h_index = i
            continue
    if not has_correct_header:
        return None
    return {
        'type': 'image structure',
        'header_type': 1,
        'header_length': 9,
        'bits_per_pixel': file[h_index + 3],
        'columns': (file[h_index + 4] * 256) + file[h_index + 5],
        'rows': (file[h_index + 6] * 256) + file[h_index + 7],
        'compression_flag': file[h_index + 8]
    }


def get_image_navigation(file: np.array, record: list[tuple]) -> Optional[dict]:
    has_correct_header = False
    h_index = None
    for h, i, _ in record:
        if h == 2:
            has_correct_header = True
            h_index = i
            continue
    if not has_correct_header:
        return None
    return {
        'type': 'image structure',
        'header_type': 2,
        'header_length': 50,
        'projection_name': ''.join(chr(v) for v in file[h_index + 3:h_index + 35]),
        'column_scaling_factor': signed(sum([file[35 + k] * (256 ** (3 - k)) for k in range(4)]), 32),
        'line_scaling_factor': signed(sum([file[39 + k] * (256 ** (3 - k)) for k in range(4)]), 32),
        'column_offset': signed(sum([file[43 + k] * (256 ** (3 - k)) for k in range(4)]), 32),
        'line_offset': signed(sum([file[47 + k] * (256 ** (3 - k)) for k in range(4)]), 32)
    }


def get_sequencing(file: np.array, record: list[tuple]) -> Optional[dict]:
    has_correct_header = False
    h_index = None
    for h, i, _ in record:
        if h == 128:
            has_correct_header = True
            h_index = i
            continue
    if not has_correct_header:
        return None
    return {
        'type': 'sequencing Header',
        'header_type': 128,
        'header_length': 17,
        'image_id': sum([file[h_index + 3 + k] * (256 ** (1 - k)) for k in range(2)]),
        'sequence': sum([file[h_index + 5 + k] * (256 ** (1 - k)) for k in range(2)]),
        'start_column': sum([file[h_index + 7 + k] * (256 ** (1 - k)) for k in range(2)]),
        'start_row': sum([file[h_index + 9 + k] * (256 ** (1 - k)) for k in range(2)]),
        'max_segments': sum([file[h_index + 11 + k] * (256 ** (1 - k)) for k in range(2)]),
        'max_columns': sum([file[h_index + 13 + k] * (256 ** (1 - k)) for k in range(2)]),
        'max_rows': sum([file[h_index + 15 + k] * (256 ** (1 - k)) for k in range(2)]),
    }


def write_image_file(filename: str, file: np.array, record: list[tuple]) -> str:
    """
    Writes a file of LRIT filetype '0' to the Imagery directory

    :param filename: the name of the file to write; directory and extension are determined automatically
    :param file: numpy array of type uint8 representing the bytes of the file
    :param record: list of LRIT headers for the file
    :return: string indicating success, skip, or specific error
    """
    structure = get_image_structure(file, record)
    sequence = get_sequencing(file, record)
    if structure is None:
        return f'bad input for file {filename}: Missing or broken Image Structure Header.'
    if structure['bits_per_pixel'] != 8:
        return f'bad input for file {filename}: Invalid bits per pixel value'

    data_start = get_primary(file)['total_header_length']
    magic = check_magic(file[data_start:data_start + 6])
    if magic == 'gif87' or magic == 'gif89':
        if args:
            if not args.graphics:
                return 'Filetype not processed'
        timestamp = get_time_path(filename.split('-')[0])
        filename = filename.split('-')[1]
        directory = f'{dest_path}/Imagery/{timestamp}'
        makedirs(directory, exist_ok=True)
        if path.isdir(f'{directory}/{filename}.gif'):
            return 'Skipped'
        with open(f'{directory}/{filename}.gif', 'wb') as gif_file:
            gif_file.write(bytes(file[data_start:]))
        if directory not in manifest_updates:
            manifest_updates.append(directory)
        return 'Success'
    if args:
        if not args.imagery:
            return 'Filetype not processed'
    seg = filename.split('-')
    timestamp = get_time_path(seg[3].split('_')[4][1:], month=False)

    # Make image directories if needed
    makedirs(f'{dest_path}/Imagery/{timestamp}', exist_ok=True)

    # sequenced vs. un-sequenced (mesoscale) Imagery
    if sequence is None:
        shape = (structure['rows'], structure['columns'])
        file_img = np.reshape(file[data_start:], shape)
        directory = f'{dest_path}/Imagery/{timestamp}'
        file_path = f'{directory}/{seg[0]}-{seg[1]}-{seg[2]}-{seg[3].split("_")[0]}-{seg[3].split("_")[4][1:]}.png'
        if path.exists(file_path):
            return 'Skipped'
        else:
            if directory not in manifest_updates:
                manifest_updates.append(directory)
            cv.imwrite(file_path, file_img)
            return 'Success'
    else:
        # image is a multipart file, get and assemble all LRIT files that make up this image
        structure = [structure]
        sequence = [sequence]
        file = [file]
        directory = f'{dest_path}/Imagery/{timestamp}'
        try:
            file_path = f'{directory}/{seg[0]}-{seg[1]}-{seg[2]}-{seg[3].split("_")[0]}-{seg[3].split("_")[3][1:]}-{sequence[0]["image_id"]}.png'
        except IndexError:
            return f'Index Error'
        search_name = f'{"_".join(filename.split("_")[:-1])}'
        chunks_towrite = [*range(sequence[0]['max_segments'])]

        # read image and metadata if it exists
        if path.exists(file_path):
            metadata = ImageData.read_comment(Image(file_path))
            chunks = [int(s) for s in metadata.split(',')]
            if sequence[0]['sequence'] in chunks:
                return f'Skipped'
            else:
                pass
            for c in chunks:
                if c in chunks_towrite:
                    chunks_towrite.remove(c)
            img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
        else:
            if directory not in manifest_updates:
                manifest_updates.append(directory)
            chunks = []
            img = np.zeros((sequence[0]['max_rows'], sequence[0]['max_columns']), dtype=np.uint8)
        chunks_found = []

        # find other LRIT files in sequence matching this image
        for c in chunks_towrite:
            name = f'{source_path}/{search_name}_{str(c).rjust(3, "0")}.lrit'
            if name == f'{filename}.lrit':
                break
            if path.exists(name):
                new_file = np.fromfile(name, dtype=np.uint8)
                new_record = get_header_record(new_file)
                new_structure = get_image_structure(new_file, new_record)
                new_sequence = get_sequencing(new_file, new_record)
                if new_structure is None:
                    continue
                if new_sequence is None:
                    continue
                file.append(new_file)
                structure.append(new_structure)
                sequence.append(new_sequence)
                chunks_found.append(new_sequence['sequence'])

        # use file data to assemble image
        for struct, seq, file_ in zip(structure, sequence, file):
            if seq['sequence'] not in chunks:
                data_start = get_primary(file_)['total_header_length']
                shape = (struct['rows'], struct['columns'])
                file_img = np.reshape(file_[data_start:], shape)
                chunks.append(seq['sequence'])
                start_row = seq['start_row']
                end_row = start_row + struct['rows']
                img[start_row:end_row][:] = file_img

        # finally write image and add metadata
        cv.imwrite(file_path, img)
        comment = ','.join([str(c) for c in chunks])
        ImageData.modify_comment(Image(file_path), comment)
    return 'Success'


def write_text_file(filename: str, file: np.array) -> str:
    """
    Writes a file of LRIT filetype '2' to the Text directory. Assumes files are uncompressed plaintext

    :param filename: the name of the file to write; directory and extension are determined automatically
    :param file: numpy array of type uint8 representing the bytes of the file
    :return: string indicating success, skip, or specific error
    """
    directory = None
    file_path = f'{dest_path}/{filename}.txt'
    if filename[:23] != 'GOES_EAST_Admin_message':
        parts = filename.split('_')
        timestamp = get_time_path(parts[4])
        directory = f'{dest_path}/Text/{timestamp}'
        file_path = f'{directory}/{filename}.txt'
        if path.exists(file_path):
            return 'Skipped'

        # make date directories if necessary
        makedirs(f'{dest_path}/Text/{timestamp}', exist_ok=True)
    elif path.exists(file_path):
        return 'Skipped'

    primary = get_primary(file)
    data_start = primary['total_header_length']
    with open(file_path, 'w', encoding='utf-8') as text_file:
        text_file.write(''.join([chr(i) for i in file[data_start:]]))
    if directory is not None and directory not in manifest_updates:
        manifest_updates.append(directory)
    return 'Success'


def write_dcs_file(filename: str, file: np.array) -> str:
    """
    Writes a file of LRIT filetype '130' to the DCS directory

    :param filename: the name of the file to write; directory and extension are determined automatically
    :param file: numpy array of type uint8 representing the bytes of the file
    :return: string indicating success, skip, or specific error
    """
    data_start = get_primary(file)['total_header_length']
    block_id = 0
    offset = 64 + data_start
    while block_id != 1:
        block_length = (file[offset + 2] << 8) | file[offset + 1]
        block_id = file[offset]
        if block_id != 1:
            offset += block_length
    timestamp = bcd_to_time_path(file[offset + 0x0C:offset + 0x13])
    directory = f'{dest_path}/DCS/{timestamp}'
    if path.exists(f'{directory}/{filename}.csv'):
        return 'Skipped'

    # make date directories if necessary
    makedirs(directory, exist_ok=True)

    dcs_header = {'filename': ''.join([chr(c) for c in file[data_start + 0:data_start + 32]]),
                  'file_size': int(''.join([chr(c & 0x7F) for c in file[data_start + 32: data_start + 40]])),
                  'file_source': ''.join([chr(c) for c in file[data_start + 40:data_start + 44]]),
                  'file_type': ''.join([chr(c) for c in file[data_start + 44:data_start + 48]])}
    # I don't check the CRC because I'm a chad like that.
    offset = 64 + data_start
    with open(f'{directory}/{filename}.csv', 'w', encoding='utf-8') as DCS_file:
        DCS_file.write(
            'size,seq_num,data_rate,platform,parity_error,ARM_flags,corrected_address,carrier_start,message_end,signal_strength,freq_offset,phs_noise,modulation_index,good_phs,channel,spacecraft,source_code,source_secondary,data,crc_ok\n'
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
                          ''.join([chr(v & 0x7F) for v in block[0x23:0x25]]),  # source_code
                          ''.join([chr(v & 0x7F) for v in block[0x25:0x27]]),  # secondary_source
                          ''.join([f'{v & 0x7F}|' for v in block[0x27:-2]])  # data
                          ]
                dcp_crc16 = (block[-1] << 8) | block[-2]
                calc_crc = crc_hqx(block[:-2], 0xFFFF)

                if dcp_crc16 == calc_crc:
                    output.append(1)
                else:
                    output.append(0)
                [DCS_file.write(f'{key},') for key in output]
                DCS_file.write('\n')
            elif block_id == 2:
                # missed block
                pass
            offset += block_length
    if directory not in manifest_updates:
        manifest_updates.append(directory)
    return 'Success'


def write_compressed_file(filename: str, file: np.array) -> str:
    """
    Decompresses and writes a file of compressed ZIP file to the appropriate directory(ies)

    :param filename: the name of the file to write; directory and extension are determined automatically
    :param file: numpy array of type uint8 representing the bytes of the file
    :return: string indicating success, skip, or specific error
    """
    file_dir = f'{dest_path}/tmp/{filename}'
    makedirs(file_dir, exist_ok=True)
    data_start = get_primary(file)['total_header_length']
    status = 'Success'
    with open(f'{file_dir}.zip', 'wb') as zip_file:
        zip_file.write(bytes(file[data_start:]))
    with ZipFile(f'{file_dir}.zip') as zip_ref:
        zip_ref.extractall(file_dir)
    for unpacked in listdir(file_dir):
        unpacked_dir = f'{file_dir}/{unpacked}'
        _, ext = unpacked.split('.')
        if args:
            if ext == 'TXT' and not args.text:
                status = 'Filetype not processed'
            elif not args.graphics:
                status = 'Filetype not processed'
        timestamp = get_time_path(unpacked.split('_')[4])
        output_dir = f'{dest_path}/Text/{timestamp}' if ext == 'TXT' else f'{dest_path}/Imagery/{timestamp}'
        makedirs(output_dir, exist_ok=True)
        if path.exists(f'{output_dir}/{unpacked}'):
            remove(unpacked_dir)
            continue
        if output_dir not in manifest_updates:
            manifest_updates.append(output_dir)
        rename(unpacked_dir, f'{output_dir}/{unpacked}')
    remove(f'{file_dir}.zip')
    rmdir(file_dir)
    return status


def write_data(filename: str, file: np.array) -> str:
    """
    Writes an LRIT file to an appropriate directory, automatically determining filetype and compression.

    :param filename: the name of the file to write; directory and extension are determined automatically
    :param file: numpy array of type uint8 representing the bytes of the file
    :return: string indicating success, skip, or specific error
    """
    primary = get_primary(file)
    file_type = primary['file_type_code']
    data_start = primary['total_header_length']
    magic = check_magic(file[data_start:data_start + 6])
    if magic == 'zip':
        if args:
            if not (args.text or args.graphics):
                return 'Filetype not processed'
        return write_compressed_file(filename, file)
    if file_type == 0:
        if args:
            if not (args.imagery or args.graphics):
                return 'Filetype not processed'
        # Image Data File (filename generated automatically, passed argument used for printing warnings)
        return write_image_file(filename, file, get_header_record(file))
    elif file_type == 2:
        if args:
            if not args.text:
                return 'Filetype not processed'
        # Alphanumeric Text File
        return write_text_file(filename, file)
    elif file_type == 130:
        if args:
            if not args.dcs:
                return 'Filetype not processed'
        # DCS File
        return write_dcs_file(filename, file)
    else:
        return 'Unrecognized file type'


def write_manifest(directory: str):
    encoder = JSONEncoder(indent=4, separators=(',', ':'))
    split = directory.split('/')
    category = split[-4]
    json_arr = {'files': []}
    for file in [*scandir(directory)]:
        meta = {'filename': file.name, 'filesize': file.stat().st_size, 'timestamp': None, 'type': None, 'properties': {}}

        if category == 'Imagery' and file.name[:2] == 'OR':
            split = file.name.split('-')
            meta['type'] = 'abi'
            try:
                meta['timestamp'] = int(datetime.strptime(split[4].split('.')[0][:-1], '%Y%j%H%M%S').timestamp())
            except IndexError:
                split_2 = split[3].split('_')
                meta['timestamp'] = int(datetime.strptime(split_2[2][1:-1], '%Y%j%H%M%S').timestamp())
            meta['properties']['imageType'] = 'fulldisk' if split[2][-1] == 'F' else 'mesoscale'
            meta['properties']['channel'] = None if split[2][:4] != 'CMIP' else split[3][3:]
            meta['properties']['productID'] = split[2][:-1] if split[2][-1] == 'F' else split[2][:-2]

        if category == 'Imagery' and not (file.name[:1] in [*'OZ']):  # Top of NOAA Hour Graphic
            split = directory.split('/')
            meta['type'] = 'hourly'
            meta['timestamp'] = int(
                datetime.strptime(''.join([split[-3], split[-2], split[-1]]), '%Y%j%H').timestamp())

        if file.name[:1] in [*'AZ']:
            split = file.name.split('_')
            if file.name[:1] == 'A':
                meta['type'] = 'text'
            else:
                meta['type'] = 'imagery'
            meta['timestamp'] = int(datetime.strptime(split[4][:-1], '%Y%m%d%H%M%S').timestamp())
            meta['properties']['messageSequenceNumber'] = int(split[5][0:6])

            try:
                meta['properties']['awds'] = lookup['awds'][split[1][0:10]]
            except KeyError:
                logger.debug(f'{split[1][0:10]} not found in AWDS lookup')
                meta['properties']['awds'] = lookup['awds']['default']

            try:
                meta['properties']['nnn'] = lookup['nnn'][split[5][9:12]]
            except KeyError:
                logger.debug(f'{split[5][9:12]} not found in NNN lookup')
                meta['properties']['nnn'] = lookup['nnn']['default']

            try:
                meta['properties']['cccc'] = lookup['cccc'][split[1][6:10]]
            except KeyError:
                logger.debug(f'{split[1][6:10]} not found in CCCC lookup')
                meta['properties']['cccc'] = lookup['cccc']['default']

            t1, t2, a1, a2, i1, i2 = split[1][0:6].upper()
            ii = str(int(i1 + i2)).rjust(2, '0')
            try:
                table_a = lookup['386']['a'][t1]
            except KeyError:
                logger.error(f'{t1} not found in 386-a lookup')
                continue
            meta['properties']['TTAA'] = {
                'generalDataType': table_a['generalDataType']
            }

            # logger.debug(f'[{split[1][0:10]}] {table_a}')

            if table_a['T2'] is not None:
                try:
                    meta['properties']['TTAA'] |= lookup['386'][table_a['T2']][t2]
                except KeyError:
                    logger.warn(f'[{split[1][0:10]}] Key \'{t2}\' not found in table {table_a["T2"]}')

            if table_a['A1'] == table_a['A2']:
                try:
                    if isinstance(table_a['A1'], list):
                        pass
                    else:
                        meta['properties']['TTAA'] |= lookup['386'][table_a['A1']][a1 + a2]
                except KeyError:
                    logger.warn(f'[{split[1][0:10]}] Key \'{a1 + a2}\' not found in table {table_a["A1"]}')
            elif table_a['A1'] == 'C6' or table_a['A1'] == 'C7':
                try:
                    table = copy.copy(lookup['386'][table_a['A1']][t1 + t2 + a1])
                    try:
                        meta['properties']['TTAA'] |= lookup['386'][table_a['A2']][a2]
                        table_index = 0
                        while (not table['ii'][0] <= ii <= table['ii'][1]) and table_index <= 2:
                            table = copy.copy(lookup['386'][table_a['A1']][f'{t1}{t2}{a1}-{table_index}'])
                            table_index += 1
                        meta['properties']['TTAA'] |= table
                    except KeyError:
                        logger.warn(f'[{split[1][0:10]}] invalid ii value for TTAAii {t1 + t2 + a1 + a1}{ii}')
                except KeyError:
                    logger.warn(f'[{split[1][0:10]}] Key \'{t1 + t2 + a1}\' not found in table {table_a["A1"]}')
            elif table_a['A1'] is not None:
                try:
                    meta['properties']['TTAA'] |= lookup['386'][table_a['A1']][a1]
                except KeyError:
                    logger.warn(f'[{split[1][0:10]}] Key \'{a1}\' not found in table {table_a["A1"]}')
                try:
                    meta['properties']['TTAA'] |= lookup['386'][table_a['A2']][a2]
                except KeyError:
                    logger.warn(f'[{split[1][0:10]}] Key \'{a2}\' not found in table {table_a["A2"]}')

            if table_a['ii'] is not None:
                try:
                    if isinstance(lookup['386'][table_a['ii']][ii], list):
                        for key in lookup['386'][table_a['ii']].keys():
                            table = copy.copy(lookup['386'][table_a['ii']][key])
                            if table['TT'] == t1 + t2 and table['ii'][0] <= ii <= table['ii'][1]:
                                meta['properties']['TTAA'] |= table
                                break
                    else:
                        try:
                            meta['properties']['TTAA'] |= lookup['386'][table_a['ii']][ii]
                        except KeyError:
                            logger.warn(f'invalid ii value for TTAAii {t1 + t2 + a1 + a2}{ii}')
                except KeyError:
                    logger.warn(f'[{split[1][0:10]}] invalid ii value for TTAAii {t1 + t2 + a1 + a1}{ii}')
        if category == 'DCS':
            split = file.name.split('-')
            meta['type'] = 'dcs'
            meta['timestamp'] = int(datetime.strptime(split[1], '%y%j%H%M%S').timestamp())
        json_arr['files'].append(meta)
    with open(f'{directory}/manifest.json', 'w') as manifest:
        manifest.write(encoder.encode(json_arr))


if __name__ == '__main__':

    from colorama import init
    init()

    parser = argparse.ArgumentParser(prog='lritproc',
                                     usage='%(prog)s [options] in_path out_path',
                                     description='process GOES LRIT files',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('in_path',
                        type=str,
                        help='path to directory containing LRIT files', )
    parser.add_argument('out_path',
                        type=str,
                        help='path to directory to put output files', )
    output_group = parser.add_argument_group('output arguments')
    remove_help_group = parser.add_argument_group('deletion options')
    remove_group = remove_help_group.add_mutually_exclusive_group()
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
    parser.add_argument('--debug',
                        help='prints debug information',
                        action='store_true',
                        required=False)
    parser.add_argument('--progress',
                        help='creates a progress bar while processing files',
                        action='store_true',
                        required=False)
    parser.add_argument('--mkdir',
                        help='make output directory if it doesn\'t exist (not recommended). lazy bastard',
                        action='store_true',
                        required=False)
    remove_group.add_argument('--remove',
                              help='remove all specified and successfully processed LRIT files',
                              action='store_true',
                              required=False)
    remove_group.add_argument('--remove-unsafe',
                              help='remove all specified and processed LRIT files',
                              action='store_true',
                              required=False)
    remove_group.add_argument('--remove-nuclear',
                              help='remove all LRIT files regardless of filetypes specified',
                              action='store_true',
                              required=False)
    args = parser.parse_args()
    if not path.isdir(args.in_path):
        print(f'Source path \'{args.in_path}\' does not exist')
        exit(-1)
    if not path.isdir(args.out_path):
        if not args.mkdir:
            print(f'Destination path \'{args.out_path}\' does not exist')
            exit(-1)
        else:
            makedirs(args.out_path)
    if args.all:
        args.text = True
        args.imagery = True
        args.graphics = True
        args.dcs = True
    if not (args.text | args.imagery | args.graphics | args.dcs):
        logger.error('No output file types specified')
        exit(-1)

    source_path = str(args.in_path)
    dest_path = str(args.out_path)
    start = perf_counter()

    logger.set_file(rf'{dest_path}/log.txt')

    logger.set_level('DEBUG', args.debug and not args.quiet and not args.progress)
    logger.set_level('INFO', args.verbose and not args.quiet)
    logger.set_level('WARN', (args.verbose | args.debug) and not args.quiet and not args.progress)
    logger.set_level('ERROR', not args.quiet and not args.progress)
    logger.set_level('SUCCESS', not args.quiet and not args.progress)

    tick = 0
    files = 0
    files_reset = 0

    lrit_files = []

    hist = [0]

    for f in listdir(source_path):
        if f.split('.')[-1].lower() == 'lrit':
            lrit_files.append(f)

    file_count = len(lrit_files)

    logger.info(f'{file_count} lrit files found')

    logger.set_level('INFO', args.verbose and not args.quiet and not args.progress)

    for lrit_file in lrit_files:
        f_start = perf_counter()
        f = np.fromfile(f'{source_path}/{lrit_file}', np.uint8)
        result = write_data(lrit_file.split('.')[0], f)
        f_end = perf_counter()

        # conditionally remove LRIT files based on remove arguments
        if args.remove_nuclear:
            remove(f'{source_path}/{lrit_file}')
        if args.remove_unsafe:
            if result != 'Filetype not processed':
                remove(f'{source_path}/{lrit_file}')
        if args.remove:
            if result in ('Success', 'Skipped'):
                remove(f'{source_path}/{lrit_file}')

        if result in ('Success', 'Skipped', 'Filetype not processed'):
            logger.debug(f'Processed {lrit_file:^82} &2| &3{(f_end - f_start) * 1000:5.1f}ms &2| &3{result}')
        else:
            logger.error(f'{lrit_file:^82} &2| &c{(f_end - f_start) * 1000:5.1f}ms &2| &c{result}')
        files += 1
        files_reset += 1

        if (perf_counter() - start) > tick:
            if args.progress:
                if len(hist) >= 15:
                    eta = int((file_count - files) / (sum(hist[-15:]) / 15))
                    logger.progress_bar(30, files, len(lrit_files), 'Processing files', f'{files} / {len(lrit_files)} [{eta // 60:02}:{eta % 60:02} remaining] {lrit_file:30}')
                else:
                    logger.progress_bar(30, files, len(lrit_files), 'Processing files', f'{files} / {len(lrit_files)} [calculating... ] {lrit_file:30}')
                hist.append(files_reset / 0.2)
                files_reset = 0
                tick += 0.2
            else:
                logger.info(f'{files:5} files processed in {tick // 60:02}:{tick % 60:02} [{files_reset:4} in the last second]')
                files_reset = 0
                tick += 1

    if args.progress:
        logger.progress_bar(30, files, len(lrit_files), 'Processing files  ', f'{files:5} / {len(lrit_files):5} [finished]')
        print()

    logger.set_level('DEBUG', args.debug and not args.quiet)
    logger.set_level('INFO', args.verbose and not args.quiet)
    logger.set_level('WARN', (args.verbose | args.debug) and not args.quiet)
    logger.set_level('ERROR', not args.quiet)
    logger.set_level('SUCCESS', not args.quiet)

    end = perf_counter()
    logger.success(f'All LRIT files successfully processed in {end - start:.4f} seconds')
    if path.isdir(f'{dest_path}/tmp'):
        rmdir(f'{dest_path}/tmp')

    start = perf_counter()
    tick = 0
    manifests = 0
    total_manifests = len(manifest_updates)

    logger.info(f'{total_manifests} manifests to generate')

    logger.set_level('DEBUG', args.debug and not args.quiet and not args.progress)
    logger.set_level('INFO', args.verbose and not args.quiet)
    logger.set_level('WARN', (args.verbose | args.debug) and not args.quiet and not args.progress)
    logger.set_level('ERROR', not args.quiet and not args.progress)
    logger.set_level('SUCCESS', not args.quiet and not args.progress)

    for manifest_dir in manifest_updates:
        f_start = perf_counter()
        write_manifest(manifest_dir)
        f_end = perf_counter()
        manifests += 1
        if args.progress:
            logger.progress_bar(30, manifests, total_manifests, 'Creating manifests', f'{manifests:3} / {total_manifests:3}')
        else:
            dir_string = manifest_dir.replace(dest_path, "").replace(r'/', '/')
            logger.info(f'Created manifest for {manifest_dir.replace(dest_path, ""):20} in {(f_end - f_start) * 1000:5.1f}ms')

    if args.progress:
        print()

    logger.set_level('DEBUG', args.debug and not args.quiet)
    logger.set_level('INFO', args.verbose and not args.quiet)
    logger.set_level('WARN', (args.verbose | args.debug) and not args.quiet)
    logger.set_level('ERROR', not args.quiet)
    logger.set_level('SUCCESS', not args.quiet)

    end = perf_counter()
    logger.success(f'All Manifests generated in {end - start:.4f} seconds')
    logger.close()
