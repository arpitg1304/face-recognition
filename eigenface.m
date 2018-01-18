close all;
clear;
clc;
% Reading the faces from 'faces' directory:
[im, person, number, subset] = readFaceImages('faces');
[lighting,subset1, persons1] = return_subset(im, subset,1, number, person);
[lighting2, subset2, persons2] = return_subset(im, subset,2, number, person);
[lighting3, subset3, persons3] = return_subset(im, subset,3, number, person);
[lighting4, subset4, persons4] = return_subset(im, subset,4, number, person);
[lighting5, subset5, persons5] = return_subset(im, subset,5, number, person);
subset6 = [subset1, subset5];
persons6 = [persons1,persons5];
superset = {subset1, subset2, subset3, subset4, subset5, subset6};
superset_p = {persons1, persons2, persons3, persons4, persons5, persons6};

%Taking inputs from the user for set to train set to test and the parameter
%d
s_j = input('Set to train:  ');
s_i = input('Set to test:  ');
s_k = input('Set the parameter d(between 1-70): ');

face_matrix1 = [];
for i = 1:size(superset{s_j},2)
    face_vector = reshape(superset{s_j}{i}, 50*50,1);
    face_matrix1 = [face_matrix1, face_vector];
end
face_matrix2 = [];
for i = 1:size(superset{s_i},2)
    face_vector2 = reshape(superset{s_i}{i}, 50*50,1);
    face_matrix2 = [face_matrix2, face_vector2];
end
mean_test_face = mean(face_matrix2,2);

mean_train_face = mean(face_matrix1,2);
norm_matrix = face_matrix1 - mean_train_face;
%Using the trick mentioned in the lecture
pseudo_cov_matrix = norm_matrix' * norm_matrix;
%[V, D] = eig(pseudo_cov_matrix);
[U, D, V] = svd(pseudo_cov_matrix);
d = 9;
d1 = 30;
eigen_vec = [];
for i = 1:s_k
    eigen_vec = [eigen_vec U(:,i)];
end
eigen_faces = norm_matrix * eigen_vec;

%creating the subplot
if s_k == 9
    figure;
    for k = 1:d
        subplot(3,3,k)
        imagesc(show_faces(eigen_faces,k))
        colormap gray
        axis off
        axis image
    end
    %saveas(figure, 'eigenfaces_d_9.png');
end

if s_k == 30
    figure;
    for k = 1:d1
        subplot(5,6,k)
        imagesc(show_faces(eigen_faces,k))
        colormap gray
        axis off
        axis image
    end
    %saveas(figure, 'eigenfaces_d_30.png');
end

trained = eigen_faces' * norm_matrix; 

full_n = [];
full_lighting = [];
full_persons= [];

projected = [];
for i = 1:size(superset{s_i},2)
    [temp_n, persons_2, projected1, dist] = check_accuracy(im, subset, s_i, trained, mean_train_face, eigen_faces, i, number, person, size(superset{s_j},2));
    full_persons = [full_persons superset_p{s_j}(temp_n)];
    projected = [projected projected1];
end

similars = full_persons == persons_2;
if sum(similars(:)) == size(similars,2)
    disp('Full Match');
end
accuracy = (sum(similars(:)) / size(similars,2)) *100;
disp('Error rate is: ');
disp(100-accuracy);

%Below code was used for reconstruction of the faces in subset 1 to 5
%reconstructing images
% figure;
% reconstructed = [];
% for i = 1:d
%     r1 = projected(i,20) .* eigen_faces(:,i);
%     reconstructed = [reconstructed, r1];
% end
% reconstructed = reconstructed + mean_train_face;
% reconstructed = sum(reconstructed,2);
% reconstructed = reshape(reconstructed,50,50);
% imagesc(reconstructed);
% colormap gray;
% axis off;
% axis image;