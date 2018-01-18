function [n, persons, projected, dist] = check_accuracy_fisher(im, subset, n, trained, mean_train_face, fisher_faces, m_a, number, person, s_set)
    [lighting, sub_test, persons] = return_subset(im, subset, n, number, person);
    dist = [];
    test_image = reshape(sub_test{m_a}, 50*50, 1);
    diff = test_image - mean_train_face;
    %new_eigen = eigenfaces - mean(eigenfaces,2);
    projected = fisher_faces' * diff;
    for i = 1:s_set
        temp = (norm(projected - trained(:,i)));
        dist = [dist temp];
    end
    [m,n] = min(dist);
end