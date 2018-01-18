clear;
clc;
%Reading files and segregating the subsets 1-5
[im, person, number, subset] = readFaceImages('faces');
[lighting,subset1, persons1] = return_subset(im, subset,1, number, person);
[lighting2, subset2, persons2] = return_subset(im, subset,2, number, person);
[lighting3, subset3, persons3] = return_subset(im, subset,3, number, person);
[lighting4, subset4, persons4] = return_subset(im, subset,4, number, person);
[lighting5, subset5, persons5] = return_subset(im, subset,5, number, person);

%Creating hybrid set of subset 1 and 5
subset6 = [subset1, subset5];
persons6 = [persons1,persons5];
superset = {subset1, subset2, subset3, subset4, subset5, subset6};
superset_p = {persons1, persons2, persons3, persons4, persons5, persons6};

%Taking inputs from user
s_j = input('Set to train:  ');
s_i = input('Set to test:  ');
s_k = input('Set the parameter c(10 or 31): ');

L = size(superset{s_j},2);

%Creating a matrix of all the faces in training set
face_matrix = [];
for i =1: size(superset{s_j},2)
    face_vector = reshape(superset{s_j}{i}, 50*50,1);
    face_matrix = [face_matrix, face_vector];
end

%Finding mean and normalizing the face matrix
mean_face = mean(face_matrix,2);

norm_matrix = face_matrix - mean_face;

%Performing PCS for dimensionality reduction
pseudo_cov = norm_matrix' * norm_matrix;
[V,D] = eig(pseudo_cov);

[D,Ind] = sort(diag(D),'descend');
V = V(:,Ind);
eig_vector = V(:, 1:s_k);

%Finding eigenfaces in higher dimension
eigen_faces = norm_matrix * eig_vector;

mean_classes= [];
for i = 1:(L/10):L
     mean_classes = [mean_classes, mean(face_matrix(:,i:i+(L/10 - 1)),2)];      
end

% Calculating Sw

S = [];
Sw = zeros(2500);

k =1;

for i = 1:10
    
    for j = 1:(L/10)
        
        S = face_matrix(:,k)- mean_classes(:,i);
        Sw = Sw+S*S';
        
        k = k+1;
    end
    
end

%Calculating Sb

Sb =zeros(2500);
for i = 1:10
    temp = (mean_classes(:,i)- mean_face); 
    Sb = Sb + (L/10)* (temp)*(temp');
end

Sb1 = eigen_faces' * Sb * eigen_faces;
Sw1 = eigen_faces' * Sw * eigen_faces;

[fisher_vect,fisher_val]  = eig(Sb1, Sw1);

[fisher_val,f_i] = sort(diag(fisher_val),'descend');
fisher_vect = fisher_vect(:,f_i);
fisher_faces = eigen_faces * fisher_vect;

%Subplot of fisher faces
if s_k == 10
    figure;
    for k = 1:9
        subplot(3,3,k)
        imagesc(show_faces(fisher_faces,k))
        colormap gray
        axis off
        axis image
    end
    %saveas(figure, 'eigenfaces_d_9.png');
end

if s_k == 31
    figure;
    for k = 1:s_k-1
        subplot(5,6,k)
        imagesc(show_faces(eigen_faces,k))
        colormap gray
        axis off
        axis image
    end
    %saveas(figure, 'eigenfaces_d_9.png');
end

trained_fisher = fisher_faces' * norm_matrix; 

full_n = [];
full_lighting = [];

full_persons= [];

projected = [];
for i = 1:size(superset{s_i},2)
    [temp_n, persons_2, projected1, dist] = check_accuracy_fisher(im, subset, s_i, trained_fisher, mean_face, fisher_faces, i, number, person, size(superset{s_j},2));
    full_persons = [full_persons superset_p{s_j}(temp_n)];
    projected = [projected projected1];
end

%Comparing obtained face numbers and the original face numbers to find the
%accuracy
similars = full_persons == persons_2;
if sum(similars(:)) == size(similars,2)
    disp('Full Match');
end
accuracy = (sum(similars(:)) / size(similars,2)) *100;
disp('Error rate is:');
disp(100-accuracy);
