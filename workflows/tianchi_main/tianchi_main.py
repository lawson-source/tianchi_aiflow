import ai_flow as af
from ai_flow import init_ai_flow_context
from ai_flow.api.ops import action_on_model_version_event
from ai_flow.model_center.entity.model_version_stage import ModelVersionEventType
from ai_flow_plugins.job_plugins.flink import set_flink_env, FlinkStreamEnv
from tianchi_executor import *
from ai_flow.workflow.status import Status
from ai_flow.workflow.control_edge import JobAction
import time


def run_tianchi_project(input_topic, output_topic, bootstrap_servers):
    af.current_graph().clear_graph()
    init_ai_flow_context()
    set_flink_env(FlinkStreamEnv())
    project_name = af.current_project_config().get_project_name()
    artifact_prefix = project_name + "."
    with af.job_config('train_job'):
        train_model = af.register_model(model_name=artifact_prefix + 'tianchi_model',
                                        model_desc='Tianchi antispam model')
        train_channel = af.train(input=[],
                                 training_processor=TrainModel(),
                                 model_info=train_model)

    with af.job_config('predict_job') as config:
        predict_input_dataset = af.register_dataset(name=artifact_prefix + 'predict_input_dataset',
                                                    uri='tianchi_input',
                                                    data_format='csv')
        predict_read_dataset = af.read_dataset(dataset_info=predict_input_dataset,
                                               read_dataset_processor=Source(input_topic, output_topic))
        predict_channel = af.predict(input=predict_read_dataset,
                                     model_info=train_model,
                                     prediction_processor=Transformer())
        write_dataset_op = af.register_dataset(name=artifact_prefix + 'predict_output_dataset',
                                               uri='tianchi_output',
                                               data_format='csv')
        af.write_dataset(input=predict_channel,
                         dataset_info=write_dataset_op,
                         write_dataset_processor=Sink())

    action_on_model_version_event(job_name='predict_job',
                                  model_name=artifact_prefix + 'tianchi_model',
                                  model_version_event_type=ModelVersionEventType.MODEL_GENERATED)

    workflow_name = af.current_workflow_config().workflow_name
    stop_workflow_executions(workflow_name)
    af.workflow_operation.submit_workflow(workflow_name)
    af.workflow_operation.start_new_workflow_execution(workflow_name)


def stop_workflow_executions(workflow_name):
    workflow_executions = af.workflow_operation.list_workflow_executions(workflow_name)
    for workflow_execution in workflow_executions:
        if workflow_execution._status == Status.RUNNING:
            af.workflow_operation.stop_workflow_execution(workflow_execution.workflow_execution_id)


if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:],"",["input_topic=","output_topic=","server="])  
    mydict = dict(opts)
    input_topic = mydict.get('--input_topic', '')
    output_topic = mydict.get('--output_topic', '')
    bootstrap_servers = mydict.get('--server', '')
    bootstrap_servers = bootstrap_servers.split(',')
    run_tianchi_project(input_topic, output_topic, bootstrap_servers)
