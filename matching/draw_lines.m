function [num_matches, img] = draw_lines(images, height, width, prob_score, index_in_gal, index_i, index_j, threshold)
  [~, ~, hei, wid] = size(prob_score);
  row_step = height / hei;
  col_step = width / wid;
  row_offset = row_step / 2;
  col_offset = col_step / 2;

  I = images{index_i};
  if ~isequal(size(I), [height, width, 3])
    I = imresize(I, [height, width]);
  end
  
  J = images{index_j};
  if ~isequal(size(J), [height, width, 3])
    J = imresize(J, [height, width]);
  end
  
  match_score = prob_score(index_i, index_j, :, :);
  match_score = reshape(match_score, [hei, wid]);
  match_index = index_in_gal(index_i, index_j, :, :);
  match_index = reshape(match_index, [hei, wid]);
  
  match_index_col = mod(match_index, wid) + 1;
  match_index_row = floor(match_index / wid) + 1;
  
  mask = match_score > threshold;
  thr_index = find(mask);
  num_matches = length(thr_index);
  
  if num_matches > 0  
    figure(1);
    new_width = round(435 * height / 337);
    half = round((new_width - 2 * width) / 2);
    img = [I, J];
    imshow(img);
    
    match_index_col = match_index_col(thr_index);
    match_index_row = match_index_row(thr_index);
    [row_i, col_i] = find(mask);
    x_i = (col_i - 1) * col_step + col_offset;
    y_i = (row_i - 1) * row_step + row_offset;
    x_j = (match_index_col - 1) * col_step + col_offset + width;
    y_j = (match_index_row - 1) * row_step + row_offset;
    x = [x_i'; x_j'];
    y = [y_i'; y_j'];
    line(x, y, 'Color', 'red');
    
    f = getframe(gca);
    img = frame2im(f);
    img = img(2:end, :, :); % 337 x 435
    img = imresize(img, [height, new_width]);
    img = img(:, half + 1 : half + 2 * width, :);
  else
    img = -1;
  end
