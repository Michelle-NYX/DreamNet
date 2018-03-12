% Loss Processing
% Mar 12 training results
clear all
close all
fileID = fopen('out.txt');
fmt = 'step: %d%s ... - loss: %f - rpn_class_loss: %f - rpn_bbox_loss: %f - mrcnn_class_loss: %f - mrcnn_bbox_loss: %f - mrcnn_mask_loss: %f';
C = textscan(fileID,fmt,'MultipleDelimsAsOne',1);
fclose(fileID);

iter = C{1};
loss = C{3};
rpn_class_loss = C{4};
rpn_bbox_loss = C{5};
mrcnn_class_loss = C{6};
mrcnn_bbox_loss = C{7};
mrcnn_mask_loss = C{8};

val_loss = [1.39136830000000;0.936763640000000;0.990963400000000;0.826807200000000;0.764838460000000;0.726978660000000;0.689516200000000;0.718414550000000];
val_iter = (0.5:1:8) * 128;

plot(iter, loss, 'LineWidth',1.5)
hold on
plot(val_iter, val_loss, 'o-', 'LineWidth',2)
xlabel('#iteration')
ylabel('loss')
legend('train loss','dev loss')
grid on

