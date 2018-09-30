import tensorflow as tf
from mnist_classifier import MnistClassifier
import mnist_loader


def main():
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    flags.DEFINE_string(
        'train_data', '',
        'taraining data file path')
    flags.DEFINE_string(
        'model_path', '',
        'model output path')
    flags.DEFINE_integer(
        'checkpoints_to_keep', 5,
        'checkpoint keep count')
    flags.DEFINE_integer(
        'keep_checkpoint_every_n_hours', 1,
        'checkpoint create ')
    flags.DEFINE_integer(
        'max_steps', 1000,
        'max trainig step')
    flags.DEFINE_integer(
        'save_checkpoint_steps', 100,
        'save checkpoint step')
    flags.DEFINE_integer(
        'batch_size', 128,
        'training batch size')

    train_data = FLAGS.train_data
    model_path = FLAGS.model_path
    checkpoints_to_keep = FLAGS.checkpoints_to_keep
    keep_checkpoint_every_n_hours = FLAGS.keep_checkpoint_every_n_hours
    max_steps = FLAGS.max_steps
    save_checkpoint_steps = FLAGS.save_checkpoint_steps
    batch_size = FLAGS.batch_size

    # load dataset
    converter = mnist_loader.Converter()
    dataset = mnist_loader.load(train_data, converter, batch_size=batch_size)
    iterator = dataset.make_one_shot_iterator()
    inputs, labels = iterator.get_next()

    # build train operation
    global_step = tf.train.get_or_create_global_step()
    model = MnistClassifier(hidden_size=512, classes=10)
    logits = model(inputs)
    loss = model.loss(logits, labels)
    train_op = model.optimize(loss)
    predict = model.predict(logits)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(predict, tf.uint8), labels), tf.float32))
    with tf.control_dependencies([train_op]):
        train_op = tf.assign_add(global_step, 1)

    # logging for tensorboard
    tf.summary.scalar('global_step', global_step)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.image('image', inputs)

    # create saver
    scaffold = tf.train.Scaffold(
        saver=tf.train.Saver(
            max_to_keep=checkpoints_to_keep,
            keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours))

    # create hooks
    hooks = []
    tf.logging.set_verbosity(tf.logging.INFO)
    metrics = {
        "global_step": global_step,
        "loss": loss,
        "accuracy": accuracy}
    hooks.append(tf.train.LoggingTensorHook(metrics, every_n_iter=100))
    hooks.append(tf.train.NanTensorHook(loss))
    if max_steps:
        hooks.append(tf.train.StopAtStepHook(last_step=max_steps))

    # training
    session = tf.train.MonitoredTrainingSession(
        checkpoint_dir=model_path,
        hooks=hooks,
        scaffold=scaffold,
        save_checkpoint_steps=save_checkpoint_steps)

    with session:
        while not session.should_stop():
            session.run([train_op, labels, predict])


main()
