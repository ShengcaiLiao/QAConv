% Demo of the QAConv matching.
% For the score matrix, probe is in rows, and gallery is in columns.
% Tested on Octave 4.4.1.

close all; clear; clc;

% modify the followings according to the test_matching.py
dataset = 'market';
prob_range = '0-1000';
model_dir = 'your_model_dir/';
score_file =[model_dir, '/', dataset, '_query_score_', prob_range, '.mat'];
img_dir = 'your_image_dir/';

threshold = 0.5;
num_match_threshold = 5;
height = 384;
width = 128;

out_dir = sprintf('%s%s%s_thr=%g/', model_dir, dataset, prob_range, threshold);
pos_out_dir = [out_dir, 'positives/'];
neg_out_dir = [out_dir, 'negatives/'];
mkdir(out_dir);
mkdir(pos_out_dir);
mkdir(neg_out_dir);

load(score_file, 'index_in_gal', 'prob_ids', 'prob_cams', 'prob_list', 'prob_score', 'score', 'fc');

% scale matching scores to make them visually more recognizable
prob_score = prob_score * 200;

num_probs = size(prob_score)(1);
prob_ids = prob_ids(1:num_probs);
prob_cams = prob_cams(1:num_probs);
prob_list = prob_list(1:num_probs);

for i = 1 : num_probs
  score(i,i) = 0;
end

images = cell(num_probs, 1);

for i = 1 : num_probs
  filename = prob_list{i};
  images{i} = imread([img_dir, filename]);
end

for i = 1 : num_probs
  sam_index = find(prob_ids == prob_ids(i) & prob_cams ~= prob_cams(i));
  num_sam = length(sam_index);
  
  for j = 1 : num_sam
    if j == i
      continue;
    end

    index_j = sam_index(j);
    file_i = prob_list{i};
    file_j = prob_list{index_j};    
    [num_matches, img] = draw_lines(images, height, width, prob_score, index_in_gal, i, index_j, threshold);
    fprintf('Probe %d: positive, score=%g, #matches=%d.\n', i, score(index_j, i), num_matches);
    
    if num_matches >= num_match_threshold
      filename = sprintf('%s%d_%.4f_%s-%s', pos_out_dir, num_matches, score(index_j, i), file_i(1:end-4), file_j);
      imwrite(img, filename);
    end
  end
          
  sam_index = find(prob_ids ~= prob_ids(i) & prob_cams ~= prob_cams(i));
  num_sam = length(sam_index);
  
  sam_score = score(sam_index, i);
  [max_score, max_index] = max(sam_score);
  
  index_j = sam_index(max_index);
  file_j = prob_list{index_j};
  
  [num_matches, img] = draw_lines(images, height, width, prob_score, index_in_gal, i, index_j, threshold);
  fprintf('\t negative, max score=%g, #matches=%d.\n', max_score, num_matches);  
  
  if num_matches >= num_match_threshold
    filename = sprintf('%s%d_%.4f_%s-%s', neg_out_dir, num_matches, max_score, file_i(1:end-4), file_j);
    imwrite(img, filename);
  end
end
