import numpy as np
import os
import sys

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)

from .cfg import CTPNConfig as cfg
from .other import Graph
from .other import normalize

# sys.path.append('.')
from lib.fast_rcnn.nms_wrapper import nms


class TextProposalGraphBuilder:
    """
        Build Text proposals into a graph.
    """

    def get_successions(self, index):
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) + 1, min(int(box[0]) + cfg.MAX_HORIZONTAL_GAP + 1, self.im_size[1])):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results
        return results

    def get_precursors(self, index):
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) - 1, max(int(box[0] - cfg.MAX_HORIZONTAL_GAP), 0) - 1, -1):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results
        return results

    def is_succession_node(self, index, succession_index):
        precursors = self.get_precursors(succession_index)
        if self.scores[index] >= np.max(self.scores[precursors]):
            return True
        return False

    def meet_v_iou(self, index1, index2):
        def overlaps_v(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            y0 = max(self.text_proposals[index2][1], self.text_proposals[index1][1])
            y1 = min(self.text_proposals[index2][3], self.text_proposals[index1][3])
            return max(0, y1 - y0 + 1) / min(h1, h2)

        def size_similarity(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            return min(h1, h2) / max(h1, h2)

        return overlaps_v(index1, index2) >= cfg.MIN_V_OVERLAPS and \
               size_similarity(index1, index2) >= cfg.MIN_SIZE_SIM

    def build_graph(self, text_proposals, scores, im_size):
        self.text_proposals = text_proposals
        self.scores = scores
        self.im_size = im_size
        self.heights = text_proposals[:, 3] - text_proposals[:, 1] + 1

        boxes_table = [[] for _ in range(self.im_size[1])]
        for index, box in enumerate(text_proposals):
            boxes_table[int(box[0])].append(index)
        self.boxes_table = boxes_table

        graph = np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)

        for index, box in enumerate(text_proposals):
            successions = self.get_successions(index)
            if len(successions) == 0:
                continue
            succession_index = successions[np.argmax(scores[successions])]
            if self.is_succession_node(index, succession_index):
                # NOTE: a box can have multiple successions(precursors) if multiple successions(precursors)
                # have equal scores.
                graph[index, succession_index] = True
        return Graph(graph)


class TextProposalConnector:
    """
        Connect text proposals into text lines
    """

    def __init__(self):
        self.graph_builder = TextProposalGraphBuilder()

    def group_text_proposals(self, text_proposals, scores, im_size):
        graph = self.graph_builder.build_graph(text_proposals, scores, im_size)
        return graph.sub_graphs_connected()

    def fit_y(self, X, Y, x1, x2):
        len(X) != 0
        # if X only include one point, the function will get line y=Y[0]
        if np.sum(X == X[0]) == len(X):
            return Y[0], Y[0]
        p = np.poly1d(np.polyfit(X, Y, 1))
        return p(x1), p(x2)

    def get_text_lines(self, text_proposals, scores, im_size):
        """
        text_proposals:boxes

        """
        # tp=text proposal
        tp_groups = self.group_text_proposals(text_proposals, scores,
                                              im_size)  ##find the text line

        text_lines = np.zeros((len(tp_groups), 8), np.float32)

        for index, tp_indices in enumerate(tp_groups):
            text_line_boxes = text_proposals[list(tp_indices)]
            num = np.size(text_line_boxes)  ##find
            X = (text_line_boxes[:, 0] + text_line_boxes[:, 2]) / 2
            Y = (text_line_boxes[:, 1] + text_line_boxes[:, 3]) / 2

            z1 = np.polyfit(X, Y, 1)
            p1 = np.poly1d(z1)

            x0 = np.min(text_line_boxes[:, 0])
            x1 = np.max(text_line_boxes[:, 2])

            offset = (text_line_boxes[0, 2] - text_line_boxes[0, 0]) * 0.5

            lt_y, rt_y = self.fit_y(text_line_boxes[:, 0],
                                    text_line_boxes[:, 1], x0 + offset,
                                    x1 - offset)
            lb_y, rb_y = self.fit_y(text_line_boxes[:, 0],
                                    text_line_boxes[:, 3], x0 + offset,
                                    x1 - offset)

            # the score of a text line is the average score of the scores
            # of all text proposals contained in the text line
            score = scores[list(tp_indices)].sum() / float(len(tp_indices))

            text_lines[index, 0] = x0
            text_lines[index, 1] = min(lt_y, rt_y)
            text_lines[index, 2] = x1
            text_lines[index, 3] = max(lb_y, rb_y)
            text_lines[index, 4] = score
            text_lines[index, 5] = z1[0]
            text_lines[index, 6] = z1[1]
            height = np.mean((text_line_boxes[:, 3] - text_line_boxes[:, 1]))
            text_lines[index, 7] = height + 2.5
        # text_lines=clip_boxes(text_lines, im_size)

        return text_lines


class TextDetector:
    def __init__(self):
        self.text_proposal_connector = TextProposalConnector()

    def detect(self, text_proposals, scores, size):
        """
        Detecting texts from an image
        :return: the bounding boxes of the detected texts
        """
        # text_proposals, scores=self.text_proposal_detector.detect(im, cfg.MEAN)
        keep_inds = np.where(scores > cfg.TEXT_PROPOSALS_MIN_SCORE)[0]
        text_proposals, scores = text_proposals[keep_inds], scores[keep_inds]

        sorted_indices = np.argsort(scores.ravel())[::-1]
        text_proposals, scores = text_proposals[sorted_indices], scores[sorted_indices]

        # nms for text proposals
        keep_inds = nms(np.hstack((text_proposals, scores)), cfg.TEXT_PROPOSALS_NMS_THRESH)
        text_proposals, scores = text_proposals[keep_inds], scores[keep_inds]

        scores = normalize(scores)

        text_lines = self.text_proposal_connector.get_text_lines(text_proposals, scores, size)

        keep_inds = self.filter_boxes(text_lines)
        text_lines = text_lines[keep_inds]

        if text_lines.shape[0] != 0:
            keep_inds = nms(text_lines, cfg.TEXT_LINE_NMS_THRESH)
            text_lines = text_lines[keep_inds]

        return text_lines

    def filter_boxes(self, boxes):
        heights = boxes[:, 3] - boxes[:, 1] + 1
        widths = boxes[:, 2] - boxes[:, 0] + 1
        scores = boxes[:, -1]
        return np.where((widths / heights > cfg.MIN_RATIO) & (scores > cfg.LINE_MIN_SCORE) &
                        (widths > (cfg.TEXT_PROPOSALS_WIDTH * cfg.MIN_NUM_PROPOSALS)))[0]
