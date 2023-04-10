from utils import *

class LandmarkContainer():
    # this class is used to store the landmarks
    # the landmarks can be 2d or 3d

    def __init__(self, landmark_type: str, landmark: dict):
        '''
        Args:
        ------
        landmark_type: str, the type of the landmarks, e.g. '2d', '3d'
        landmark: dict, the value of the landmarks

        '''
        self.type = landmark_type
        self.landmark = landmark

    @classmethod
    def load(cls, landmark_type: str, landmark_value: list, landmark_label: list):
        '''
        load the landmarks from a file with specified suffix
        the landmarks can be 2d or 3d

        Args:
        ------
        landmark_type: str, the type of the landmarks, e.g. '2d', name_format: str '3d'
        landmark_value: list, the value of the landmarks
        landmark_label: list, the label of the landmarks

        Returns:
        --------
        class instance

        '''
        landmark = {}

        landmark_label = cls.regulate_landmark_label(
            cls, landmark_label, 'sps-l')  # regulate the naming format of the landmarks as 'sps-l'

        for i in range(len(landmark_label)):
            landmark[landmark_label[i]] = landmark_value[i]

        if landmark_type == '2d':
            pass
        elif landmark_type == '3d':
            pass
        else:
            raise ValueError(
                "The type of the landmarks should be '2d' or '3d'")

        return cls(landmark_label, landmark_type)

    def regulate_landmark_label(self, names: list, name_format: str) -> list:
        '''
        rename the label name of the landmarks based on the naming rule defined in the name_format

        Args:
        ------
        name: list, the name of the landmarks with certain format
        name_format: str, 
                        the format of the name, 
                        e.g. 'r_sps', 'l_sps', l stands for left, r stands for right, sps stands for sacroiliac point
                        the name_format is the format of the output name

        Returns:
        --------
        name: list
        '''
        target_label_name = []

        template = {}

        if '-' in name_format:
            name_format = name_format.split('-')
            name_format.append('-')
            if len(name_format[0]) == 1:
                # 0 denotes side, separator, label pattern
                template['order'] = 0
                # 0 denotes lower case, i.e. 'l' 'r', 1 denotes upper case 'L' 'R'
                template['side'] = 0 if name_format[0].islower() else 1
                # 0 denotes lower case, i.e. 'sps', 1 denotes upper case 'SPS'
                template['label'] = 0 if name_format[1][0].islower() else 1

            else:
                # 1 denotes label, separator, side pattern
                template['order'] = 1
                # 0 denotes lower case, i.e. 'l' 'r', 1 denotes upper case 'L' 'R'
                template['side'] = 0 if name_format[1].islower() else 1
                # 0 denotes lower case, i.e. 'sps', 1 denotes upper case 'SPS'
                template['label'] = 0 if name_format[0][0].islower() else 1

        elif '_' in name_format:
            name_format = name_format.split('_')
            name_format.append('_')

            if len(name_format[0]) == 1:
                # 0 denotes side, separator, label pattern
                template['order'] = 0

                # 0 denotes lower case, i.e. 'l' 'r', 1 denotes upper case 'L' 'R'
                template['side'] = 0 if name_format[0].islower() else 1

                # 0 denotes lower case, i.e. 'sps', 1 denotes upper case 'SPS'
                template['label'] = 0 if name_format[1][0].islower() else 1
            else:
                # 1 denotes label, separator, side pattern
                template['order'] = 1

                # 0 denotes lower case, i.e. 'l' 'r', 1 denotes upper case 'L' 'R'
                template['side'] = 0 if name_format[1].islower() else 1

                # 0 denotes lower case, i.e. 'sps', 1 denotes upper case 'SPS'
                template['label'] = 0 if name_format[0][0].islower() else 1

        else:
            pass  # TODO: add other naming format

        # if '-' in name_format else (name_format.split('_')).append('_')

        # print(template)

        if '-' in names[0]:
            divider = '-'
        elif '_' in names[0]:
            divider = '_'
        else:
            pass  # TODO: add other naming format

        for name in names:

            if divider == '-':
                anatomy_name = name.split("-")

                if len(anatomy_name[0]) == 1 and template['order'] == 0:  # side comes first
                    anatomy_name[0] = anatomy_name[0].upper()\
                        if template['side'] == 1 else anatomy_name[0].lower()
                    anatomy_name[1] = anatomy_name[1].upper()\
                        if template['label'] == 1 else anatomy_name[1].lower()

                    target_name = anatomy_name[0] + \
                        name_format[2] + anatomy_name[1]

                # label comes first
                elif len(anatomy_name[1]) == 1 and template['order'] == 0:
                    anatomy_name[1] = anatomy_name[1].upper()\
                        if template['side'] == 1 else anatomy_name[1].lower()
                    anatomy_name[0] = anatomy_name[0].upper()\
                        if template['label'] == 1 else anatomy_name[0].lower()

                    target_name = anatomy_name[1] + \
                        name_format[2] + anatomy_name[0]

                # side comes first
                elif len(anatomy_name[0]) == 1 and template['order'] == 1:
                    anatomy_name[0] = anatomy_name[0].upper()\
                        if template['side'] == 1 else anatomy_name[0].lower()
                    anatomy_name[1] = anatomy_name[1].upper()\
                        if template['label'] == 1 else anatomy_name[1].lower()

                    target_name = anatomy_name[0] + \
                        name_format[2] + anatomy_name[1]

                # label comes first
                elif len(anatomy_name[1]) == 1 and template['order'] == 1:
                    anatomy_name[1] = anatomy_name[1].upper()\
                        if template['side'] == 1 else anatomy_name[1].lower()
                    anatomy_name[0] = anatomy_name[0].upper()\
                        if template['label'] == 1 else anatomy_name[0].lower()

                    target_name = anatomy_name[1] + \
                        name_format[2] + anatomy_name[0]
                # print(target_label_name)

            elif divider == '_':
                anatomy_name = name.split("_")
                if len(anatomy_name[0]) == 1 and template['order'] == 0:  # side comes first
                    anatomy_name[0] = anatomy_name[0].upper()\
                        if template['side'] == 1 else anatomy_name[0].lower()
                    anatomy_name[1] = anatomy_name[1].upper()\
                        if template['label'] == 1 else anatomy_name[1].lower()

                    target_name = anatomy_name[0] + \
                        name_format[2] + anatomy_name[1]

                # label comes first
                elif len(anatomy_name[1]) == 1 and template['order'] == 0:
                    anatomy_name[1] = anatomy_name[1].upper()\
                        if template['side'] == 1 else anatomy_name[1].lower()
                    anatomy_name[0] = anatomy_name[0].upper()\
                        if template['label'] == 1 else anatomy_name[0].lower()

                    target_name = anatomy_name[1] + \
                        name_format[2] + anatomy_name[0]

                # side comes first
                elif len(anatomy_name[0]) == 1 and template['order'] == 1:
                    anatomy_name[0] = anatomy_name[0].upper()\
                        if template['side'] == 1 else anatomy_name[0].lower()
                    anatomy_name[1] = anatomy_name[1].upper()\
                        if template['label'] == 1 else anatomy_name[1].lower()

                    target_name = anatomy_name[0] + \
                        name_format[2] + anatomy_name[1]

                # label comes first
                elif len(anatomy_name[1]) == 1 and template['order'] == 1:
                    anatomy_name[1] = anatomy_name[1].upper()\
                        if template['side'] == 1 else anatomy_name[1].lower()
                    anatomy_name[0] = anatomy_name[0].upper()\
                        if template['label'] == 1 else anatomy_name[0].lower()

                    target_name = anatomy_name[1] + \
                        name_format[2] + anatomy_name[0]
                # print(target_label_name)

            elif divider == 'other':  # e.g. 'sps-r'
                pass  # TODO: add other naming format

            else:
                RuntimeError(
                    'The divider is not supported yet, please check the name format')
                pass

            target_label_name.append(target_name)

        # print(target_label_name)
        return target_label_name

    def get_landmark(mode: str = 'default') -> dict:
        '''
        get the value of the landmarks for a certain mode

        Args:
        ------
        mode: str,  the order and the format of the landmarks,
                    e.g. 'synthex' stands for the synthex ways of labeling the landmarks


        Returns:
        --------
        landmarks: dict,    a dictionary with keys: 'landmarks_name', 'landmarks_values'
        '''
        if mode == 'synthex':
            template = 'FH-l'

        elif mode == 'xreg':
            pass

        elif mode == 'default':
            pass

        elif mode == 'other':
            # Define your own mode here

            pass
        else:
            print('The mode is not supported yet')
        pass

    pass
