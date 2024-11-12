import argparse
import logging
import pathlib


def main():
    parser = argparse.ArgumentParser(description='Generates directories containing symlinks to ImageNet-C'
                                                 ' images grouped by severity.')
    parser.add_argument('-s', '--source_dir', type=pathlib.Path, help='Path to the original ImageNet-C data.')
    parser.add_argument('-t', '--target_dir', type=pathlib.Path, help='Path to the target directory.')
    args = parser.parse_args()
    # setup logging
    logging.basicConfig(
        format=(
            '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] ' '%(message)s'
        ),
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        force=True,
    )
    logging.info('Configured logging')
    #
    source_dir = args.source_dir
    target_dir = args.target_dir
    for distortion_path in source_dir.iterdir():
        distortion = distortion_path.name
        for source_severity_path in distortion_path.iterdir():
            severity_level = source_severity_path.name
            logging.info(f'Processing distortion: {distortion} severity: {severity_level}')
            target_severity_path = target_dir / severity_level
            for source_class_path in source_severity_path.iterdir():
                class_name = source_class_path.name
                logging.info(f'\tclass: {class_name}')
                target_class_path = target_severity_path / class_name
                target_class_path.mkdir(parents=True, exist_ok=True)
                for image_path in source_class_path.iterdir():
                    target_link_path = target_class_path / f'{distortion}_{image_path.name}'
                    target_link_path.symlink_to(image_path)
    logging.info(f'Symlink generation completed')


if __name__ == '__main__':
    main()
