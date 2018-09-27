"""
Functions for dealing with data input and output.

"""

import os
import gzip
import logging
import numpy as np
import struct
from random import shuffle

logger = logging.getLogger(__name__)


#-----------------------------------------------------------------------------#
#                            GENERAL I/O FUNCTIONS                            #
#-----------------------------------------------------------------------------#

def smart_open(filename, mode=None):
    """Opens a file normally or using gzip based on the extension."""
    if os.path.splitext(filename)[-1] == ".gz":
        if mode is None:
            mode = "rb"
        return gzip.open(filename, mode)
    else:
        if mode is None:
         mode = "r"
        return open(filename, mode)

def read_kaldi_ark_from_scp(uid, offset, batch_size, buffer_size, scp_fn, ark_base_dir=""):
    """
    Read a binary Kaldi archive and return a dict of Numpy matrices, with the
    utterance IDs of the SCP as keys. Based on the code:
    https://github.com/yajiemiao/pdnn/blob/master/io_func/kaldi_feat.py

    Parameters
    ----------
    ark_base_dir : str
        The base directory for the archives to which the SCP points.
    """

    ark_dict = {}
    totframes = 0
    lines = 0
    with open(scp_fn) as f:
        for line in f:
            lines = lines + 1
            if lines<=uid:
                continue
            if line == "":
                continue
            utt_id, path_pos = line.replace("\n", "").split()
            ark_path, pos = path_pos.split(":")
            if 'bagchid' in ark_path:
                ark_path_list = ark_path.split('/')
                ark_path = os.path.join(*ark_path_list[-3:])
            # print(ark_base_dir, ark_path)
            ark_path = os.path.join(ark_base_dir, ark_path)
            ark_read_buffer = smart_open(ark_path, "rb")
            ark_read_buffer.seek(int(pos),0)
            header = struct.unpack("<xcccc", ark_read_buffer.read(5))
            #assert header[0] == "B", "Input .ark file is not binary"
            rows = 0
            cols = 0
            m,rows = struct.unpack("<bi", ark_read_buffer.read(5))
            n,cols = struct.unpack("<bi", ark_read_buffer.read(5))
            tmp_mat = np.frombuffer(ark_read_buffer.read(rows*cols*4), dtype=np.float32)
            if len(tmp_mat) != rows * cols:
                return {}, lines
            utt_mat = np.reshape(tmp_mat, (rows, cols))
            #utt_mat_list=utt_mat.tolist()
            ark_read_buffer.close()
            ark_dict[utt_id] = utt_mat
            totframes += rows
            if totframes>=(batch_size*buffer_size-offset):
                break

    return ark_dict,lines

def kaldi_write_mats(ark_path, utt_id, utt_mat):
    ark_write_buf = smart_open(ark_path, "ab")
    utt_mat = np.asarray(utt_mat, dtype=np.float32)
    batch, rows, cols = utt_mat.shape
    ark_write_buf.write(struct.pack('<%ds'%(len(utt_id)), utt_id))
    ark_write_buf.write(struct.pack('<cxcccc', b' ',b'B',b'F',b'M',b' '))
    ark_write_buf.write(struct.pack('<bi', 4, rows))
    ark_write_buf.write(struct.pack('<bi', 4, cols))
    ark_write_buf.write(utt_mat)

def load_utterance_locations(data_dir, frame_file):

    locations = {}

    with open(os.path.join(data_dir, frame_file)) as f:
        for line in f:
            utterance_id, path = line.replace("\n", "").split()
            path, location = path.split(":")
            ark_path = os.path.join(data_dir, path)
            locations[utterance_id] = int(location)

    return locations, ark_path

def read_mat(buff, byte):
    buff.seek(byte, 0)
    header = struct.unpack("<xcccc", buff.read(5))
    m, rows = struct.unpack("<bi", buff.read(5))
    n, cols = struct.unpack("<bi", buff.read(5))
    tmp_mat = np.frombuffer(buff.read(rows * cols * 4), dtype=np.float32)
    return np.reshape(tmp_mat, (rows, cols))

class DataLoader:
    """ Class for loading features and labels from file into a buffer, and batching. """

    def __init__(self,
            base_dir,
            in_frame_file,
            out_frame_file,
            batch_size,
            buffer_size,
            context,
            out_frame_count,
            shuffle):

        """ Initialize the data loader including filling the buffer """
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.context = context
        self.out_frame_count = out_frame_count
        self.shuffle = shuffle
        
        self.uid = 0
        self.offset = 0

        in_locations, self.in_ark_path = load_utterance_locations(base_dir, in_frame_file)
        out_locations, self.out_ark_path = load_utterance_locations(base_dir, out_frame_file)

        self.locations = []
        for key in in_locations:
            self.locations.append({'id':key, 'in_byte': in_locations[key], 'out_byte': out_locations[key]})

    def read_mats(self):
        """ Read features from file into a buffer """
        #Read a buffer containing buffer_size*batch_size+offset
        #Returns a line number of the scp file

        in_ark_dict = {}
        out_ark_dict = {}
        totframes = 0

        in_ark_buffer = smart_open(self.in_ark_path, "rb")
        out_ark_buffer = smart_open(self.out_ark_path, "rb")
        while totframes < self.batch_size * self.buffer_size - self.offset and self.uid < len(self.locations):
            in_mat = read_mat(in_ark_buffer, self.locations[self.uid]['in_byte'])
            out_mat = read_mat(out_ark_buffer, self.locations[self.uid]['out_byte'])

            in_ark_dict[self.locations[self.uid]['id']] = in_mat
            out_ark_dict[self.locations[self.uid]['id']] = out_mat

            totframes += len(in_mat)
            self.uid += 1

        in_ark_buffer.close()
        out_ark_buffer.close()

        return in_ark_dict, out_ark_dict

    def _fill_buffer(self):
        """ Read data from files into buffers """

        # Read data
        in_frame_dict, out_frame_dict = self.read_mats()

        if len(in_frame_dict) == 0:
            self.empty = True
            return

        ids = sorted(in_frame_dict.keys())

        if not hasattr(self, 'offset_in_frames'):
            self.offset_in_frames = np.empty((0, in_frame_dict[ids[0]].shape[1]), np.float32)

        if not hasattr(self, 'offset_out_frames'):
            self.offset_out_frames = np.empty((0, out_frame_dict[ids[0]].shape[1]), np.float32)

        # Create frame buffers
        in_frames = [in_frame_dict[i] for i in ids]
        in_frames = np.vstack(in_frames)
        in_frames = np.concatenate((self.offset_in_frames, in_frames), axis=0)

        out_frames = [out_frame_dict[i] for i in ids]
        out_frames = np.vstack(out_frames)
        out_frames = np.concatenate((self.offset_out_frames, out_frames), axis=0)

        # Put one batch into the offset frames
        cutoff = self.batch_size * self.buffer_size
        if in_frames.shape[0] >= cutoff:
            self.offset_in_frames = in_frames[cutoff:]
            in_frames = in_frames[:cutoff]
            self.offset_out_frames = out_frames[cutoff:]
            out_frames = out_frames[:cutoff]
            
            self.offset = self.offset_in_frames.shape[0]

        in_frames = np.pad(
            array     = in_frames,
            pad_width = ((self.context + self.out_frame_count // 2,),(0,)),
            mode      = 'edge')

        # Generate a random permutation of indexes
        if self.shuffle:
            self.indexes = np.random.permutation(out_frames.shape[0] - self.out_frame_count)
        else:
            self.indexes = np.arange(out_frames.shape[0] - self.out_frame_count)

        self.in_frame_buffer = in_frames
        self.out_frame_buffer = out_frames

    def batchify(self, shuffle_batches=False, include_deltas=False):
        """ Make a batch of frames and senones """

        batch_index = 0
        self.reset(shuffle_batches)

        while not self.empty:
            start = batch_index * self.batch_size
            end = min((batch_index+1) * self.batch_size, self.out_frame_buffer.shape[0])

            # Collect the data 
            try:
                in_frame_batch = np.stack((self.in_frame_buffer[i:i+self.out_frame_count+2*self.context,]
                    for i in self.indexes[start:end]), axis = 0)

                out_frame_batch = np.stack((self.out_frame_buffer[i:i+self.out_frame_count,]
                    for i in self.indexes[start:end]), axis = 0).squeeze()

                # Increment batch, and if necessary re-fill buffer
                batch_index += 1
                if batch_index * self.batch_size >= self.out_frame_buffer.shape[0]:
                    batch_index = 0
                    self._fill_buffer()

                if include_deltas:
                    yield in_frame_batch, out_frame_batch
                else:
                    yield in_frame_batch[:,:,:257], out_frame_batch

            except:
                print("Empty array...")

                # Increment batch, and if necessary re-fill buffer
                batch_index += 1
                if batch_index * self.batch_size >= self.out_frame_buffer.shape[0]:
                    batch_index = 0
                    self._fill_buffer()

    def reset(self, shuffle_batches):
        self.uid = 0
        self.offset = 0
        self.empty = False
        if shuffle_batches:
            shuffle(self.locations)

        self._fill_buffer()
