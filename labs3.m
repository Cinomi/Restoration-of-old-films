function output = labs3(path, prefix, first, last, digits, suffix)

%
% Read a sequence of images and correct the film defects. This is the file 
% you have to fill for the coursework. Do not change the function 
% declaration, keep this skeleton. You are advised to create subfunctions.
% 
% Arguments:
%
% path: path of the files
% prefix: prefix of the filename
% first: first frame
% last: last frame
% digits: number of digits of the frame number
% suffix: suffix of the filename
%
% This should generate corrected images named [path]/corrected_[prefix][number].png
%
% Example:
%
% mov = labs3('../images','myimage', 0, 10, 4, 'png')
%   -> that will load and correct images from '../images/myimage0000.png' to '../images/myimage0010.png'
%   -> and export '../images/corrected_myimage0000.png' to '../images/corrected_myimage0010.png'
%

% Your code here


% load image sequence
mov = load_sequence(path, prefix, first, last, digits, suffix);

% get the size and frame number of image sequence
[row, col, frame_num] = size(mov);

% create a video file in .avi format
% v = VideoWriter('Task1.avi','Uncompressed AVI');
% frame_duration = 15;
% v.FrameRate = frame_num/frame_duration;
% open(v);

% convert image sequence to double format
mov = im2double(mov);


% Task 1: Detection of shot cuts
% get cut scene index and write video
cut_index = sceneCuts_detection(mov, frame_num);

% Task 2, task 3, task 4 and task 5
v2 = VideoWriter('result.avi','Uncompressed AVI');
frame_duration = 15;
v2.FrameRate = frame_num/frame_duration;
open(v2);

% neighborhood size for flicker correction
neighbor_flicker = 4;

% neighborhood size for blotch correction
neighbor_num_blotch = 20;
avg_filter = [30, 40];
extend = [22, 15];

% initialize start frame index as 1
start_f = 1;

% initialize motion mask
input_mask = zeros(row, col, frame_num);

% initialize difference array
input_diff = zeros(row, col, frame_num);

% noise threshold for mask generation
threshold_mask = [0.7, 0.65];

% threshold for blotch correction
threshold_blotch = [0.1, 0.095];

% initialize input image sequence
input_mov = mov;

% initialize edges for vertical artefacts correction
edge = zeros(size(mov));

% recursive each single shot
for j = 1 : size(cut_index, 2) + 1
    if j <= size(cut_index,2)
        end_f = cut_index(j)-1;
    else 
        end_f = frame_num;
    end
  
    % global flicker remove
     output_mov= flicker_remove(input_mov,start_f, end_f, neighbor_flicker);

    if j == size(cut_index,2)+1    % vertical artefacts correction (only apply on last shot of footage)
        [output_mov] = vertical_artefacts(output_mov, edge, start_f, end_f);
    else                           % video stabilization and blotch correction for first two shot
        [output_mov] = stabilization(output_mov, start_f, end_f, 6, 10, row, col);

        [output_mask, output_diff] = motion_mask(output_mov, input_mask, input_diff, start_f, end_f, neighbor_num_blotch, threshold_mask(j), avg_filter(j), extend(j));
        [output_mov] = blotch_correction(output_mov, output_mask, output_diff, start_f, end_f, threshold_blotch(j));
        
        input_mask = output_mask;
        input_diff = output_diff;
    end
     
    input_mov = output_mov;
    start_f = end_f+1;
end

% save output image sequence
output = save_sequence(output_mov, 'result_task2345', 'footage_', 1, 3);

% write final video
for i = 1 : frame_num
    writeVideo(v2, output_mov(:,:,i));
end

close(v2);

end


%% function for scene cuts detection
% Input: 
% mov: image sequence
% frame_num: image number included in image sequence
function [cut_index] = sceneCuts_detection(mov, frame_num)

    cut_index = [];
    output_mov = mov;
    % write the first frame into the video
    % writeVideo(v, mov(:,:,1));
    
    for n = 2 : frame_num
        % get the pixel sum of current frame and next frame
        sum1 = sum(sum(mov(:,:,n-1)));
        sum2 = sum(sum(mov(:,:,n)));
        
        % get normalize frame difference
        frame_diff = sum(sum(abs(mov(:,:,n) - mov(:,:,n-1))));
        frame_diff = frame_diff / ( (sum1+sum2)/2 );
        
        % set a threshold, if difference between frames is bigger than
        % threshold, scene cut occurs.
        current_frame = mov(:,:,n);
        if (frame_diff > 0.70)
            cut_index = [cut_index, n];
            current_frame = insertText(current_frame, [20, 20], 'Scene Cut');
            output_mov(:,:,n) = rgb2gray(current_frame);
        end
        
        % write the current frame into the video
        %writeVideo(v, current_frame);
    end
    %close(v);
    a = save_sequence(output_mov, 'result_task1', 'footage_', 1, 3)
end


%% function to find neighborhood index range
% Input:
% current_f: current frame
% start_f: start frame of one shot
% end_f: end frame of one shot
% neighbor_size: neighborhood size
function [start_index, end_index]=neighbor_index(current_f, start_f, end_f, neighbor_size)
    
    % get initial start and end index of neighborhood
    neighbor_r = round(neighbor_size/2);
    start_index = current_f - neighbor_r;
    end_index = current_f + neighbor_r;
    
    % if initial start index is out of range, move backward the neighborhood 
    if start_index < start_f
        start_index = start_f;
        end_index = start_index + neighbor_size; 
    
    % if initial end index is out of range, move forward the neighborhood
    elseif end_index > end_f
        end_index = end_f;
        start_index = end_index - neighbor_size;
    end
end


%% function for global flicker correction
% Input
% input: input image sequence
% start_f: start frame within one shot
% end_f: end frame within one shot
% neighbor_size: previous or later frames number within neighborhood
function [output] = flicker_remove(input, start_f, end_f, neighbor_size)
    
    % initialize output image sequence
    output = input;

    for i = start_f : end_f
        % get neighborhood index range
        [start_index, end_index] = neighbor_index(i, start_f, end_f, neighbor_size);
        
        % get the pixel sum of neighborhood frames
        neighbor_sum = sum(input(:,:,start_index:end_index),3);
        
        % get the average of neighborhood frames
        neighbor_avg = neighbor_sum / (neighbor_size + 1);
        
        % adjuest current frame histogram according to average histogram
        output(:,:,i) = imhistmatch(input(:,:,i), neighbor_avg);
    end
end


%% function to create motion mask
% Input
% mov: input image sequence
% input_mask: input motion mask
% input_diff: input difference array
% start_f: start frame of one shot
% end_f: end frame of one shot
% neighbor_size: the number of frames within reference neighborhood 
% threshold: noise threshold
% avg_k: coefficent for average filter
% extend: coefficient for extend operation
function [output_mask, output_diff] = motion_mask(mov, input_mask, input_diff, start_f, end_f, neighbor_size, threshold, avg_k, extend)    
    % initialize difference array
    output_diff = input_diff;
    
    % initialize mask
    output_mask1 = input_mask;
    
    % compute difference between two consecutive images
    for f1 = start_f : end_f - 1
        output_diff(:,:,f1) = abs(mov(:,:,f1) - mov(:,:,f1+1));
    end
    
    % compute mask
    for f2 = start_f : end_f
        [start_index, end_index] = neighbor_index(f2, start_f, end_f, neighbor_size);
        output_mask1(:,:,f2) = sum(output_diff(:,:,start_index:end_index-1),3);
    end
    
    % smooth mask edges and eliminate any unnecessary parts
      avg_fil1 = fspecial('average', avg_k);
      output_mask2 = imfilter(output_mask1, avg_fil1);
      output_mask2(output_mask2<threshold)=0;
      output_mask2(output_mask2>threshold)=1;
      se1 = strel('disk', extend, 4);
      output_mask2 = imdilate(output_mask2, se1);

    if start_f == 1 
         avg_fil2 = fspecial('average', 40);
         mask_tem = imfilter(output_mask1, avg_fil2);
         mask_tem(mask_tem<0.92)=0;
         mask_tem(mask_tem>=0.92)=1;
         se2 = strel('disk',35,4);
         mask_tem = imdilate(mask_tem, se2);
         output_mask2(:,:,24:45) = mask_tem(:,:,24:45);
    end

    output_mask = output_mask2; 
    
end


%% function for blotch correction
% Input
% mov: input image sequence
% input_mask: input motion mask
% input_diff: input difference array
% start_f: start frame of one shot
% end_f: end frame of one shot
function [output] = blotch_correction(mov, input_mask, input_diff, start_f, end_f, threshold)
    % initialize output image sequence
    output = mov;
    blotch = input_diff; 
    
    % set a threshold to evaluate input difference array, it larger than
    % threshold, set it as blotch
    blotch(blotch>threshold)=1;
    
    % remove blotches within motion mask
    blotch = blotch .* (1 - input_mask);
    se = strel('disk', 3, 4);
    blotch = imdilate(blotch, se);
    
    for f = start_f+2 : end_f
            pre_frame = imhistmatch((output(:,:,f-2)+output(:,:,f-1))/2,output(:,:,f-1));
            output(:,:,f)=  mov(:,:,f) .* (1-blotch(:,:,f-1)) + pre_frame .* blotch(:,:,f-1);
    end
    
end


%% function for vertical artefacts correction
% Input:
% mov: input image sequence
% edge: edge information
% start_f: start frame of one shot
% end_f: end frame of one shot
function [output] = vertical_artefacts(mov, edge, start_f, end_f)
    [row, ~, ~] = size(mov);
    output = mov;
     
    % median filter
    for f = start_f:end_f
        for r=1:row
            for k=1:5
                output(r,:,f)=medfilt1(output(r,:,f),6-k);
            end
        end
    end
    
    % extract outline edge
    la_fil = fspecial('laplacian',0);
    edge(:,:,start_f:end_f) = imfilter(output(:,:,start_f:end_f), la_fil, 'replicate');
    
    % shapen edges of smoothed frames
    output= output-edge;
    output(output<0)=0;
    output(output>1)=1;
end


%% function to get the window coordinate
% Input: 
% center_r: row of center pixel
% center_c: column of center pixel
% r: radius of square window
% row_num: total number of rows in frame
% col_num: total number of columns in frame
function [top_r, bottom_r, left_c, right_c] = window_coord(center_r, center_c, r, row_num, col_num)
    % initialize coordinate
    top_r = center_r - r;
    bottom_r = center_r + r;
    left_c = center_c - r;
    right_c = center_c + r;
    
    % check boundaries
    if top_r < 1
        top_r = 1;
        bottom_r = top_r + 2 * r;
    elseif bottom_r > row_num
        bottom_r = row_num;
        top_r = bottom_r - 2 * r;
    end
    
    if left_c < 1
        left_c = 1;
        right_c = left_c + 2 * r;
    elseif right_c > col_num
        right_c = col_num;
        left_c = right_c - 2 * r;
    end
end


%% function for video stabilization
% Input:
% mov: input image sequence
% start_f: start frame of one shot
% end_f: end frame of one shot
% patch_r: radius of patch
% window_r: radius of search window
% row_num: total number of rows in frame
% col_num: total number of columns in frame
function [output] = stabilization(mov, start_f, end_f, patch_r, window_r, row_num, col_num)
    
    output = mov;

    % get first and average frame of single shot
    avg_f = sum(mov(:,:,start_f:end_f),3)/(end_f-start_f+1);   
    first_f = mov(:,:,start_f);
    
    % detect feature points of first and average frame 
    featureP_avg = detectSURFFeatures(avg_f);
    featureP_first = detectSURFFeatures(first_f);
    
    % obtain feature descriptors 
    [feature_avg, featureP_avg] = extractFeatures(avg_f, featureP_avg);
    [feature_first, ~] = extractFeatures(first_f, featureP_first);
    
    % match features of two frames
    indexPairs = matchFeatures(feature_avg, feature_first);  % indexPairs is a Px2 matrix [feature_avg, feature_first]
    matchP_avg = featureP_avg(indexPairs(:,1),:);    % match points in average frame
    
    % number of patch
    patch_num = matchP_avg.Count;       
    % position of match points
    matchP_pos = round(matchP_avg.Location);       
    for f = start_f : end_f
        current_f = mov(:,:,f);
        cor_max = 0;      
        for p = 1:patch_num
            % get patch
            [top_r_pat, bottom_r_pat, left_c_pat, right_c_pat]=window_coord(matchP_pos(p,2), matchP_pos(p,1), patch_r, row_num, col_num);
            patch = avg_f(top_r_pat:bottom_r_pat,left_c_pat:right_c_pat);            
            % get search window
            [top_r_win, bottom_r_win, left_c_win, right_c_win]=window_coord(matchP_pos(p,2), matchP_pos(p,1), window_r, row_num, col_num);
            search_window = current_f(top_r_win:bottom_r_win, left_c_win:right_c_win);            
            % correlation calculation of patch within search window
            c = normxcorr2(patch, search_window);
            c = c(2:size(c,1)-1, 2:size(c,2)-1);            
            % get maximum correlation coefficient
            if max(c(:))>cor_max
                cor_max = max(c(:));
                [max_match_r,max_match_c]=find(c==cor_max);
            end
        end              
        % shift amount calculation
        shift_r = window_r + patch_r - max_match_r(1) + 1;
        shift_c = window_r + patch_r - max_match_c(1) + 1;           
        % eliminate unreliable shift amount
        if abs(shift_r)<8 && abs(shift_c)<8
            output(:,:,f)=imtranslate(mov(:,:,f), [shift_c, shift_r]);
        end
    end

end