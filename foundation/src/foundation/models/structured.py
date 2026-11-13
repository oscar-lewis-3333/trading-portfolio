
#local JSON-schema subset validation
import math


def validate(output,schema):
    problems=[]
    def check(value,spec,path):
        kind=spec.get('type')
        if kind=='object':
            if not isinstance(value,dict):problems.append(f'{path}: not an object')
            return
            for name in spec.get('required',[]):
                if name not in value:problems.append(f'{path}.{name}: missing')
            props=spec.get('properties',{})
            for name,v in value.items():
                if name in props:
                    check(v,props[name],path+'.'+name)
                elif spec.get('additionalProperties') is False:
                    problems.append(f'{path}.{name}: unexpected')
                elif isinstance(spec.get('additionalProperties'),dict):
                    check(v,spec['additionalProperties'],path+'.'+name)
        elif kind=='array':
            if not isinstance(value,list):
                problems.append(f'{path}: not an array')
                return
            if len(value)<spec.get('minItems',0) or len(value)>spec.get('maxItems',math.inf):problems.append(f'{path}: invalid length')
            for i,v in enumerate(value):
                check(v,spec.get('items',{}),f'{path}[{i}]')
        elif kind in {'integer','number'}:
            valid=not isinstance(value,bool) and isinstance(value,int if kind=='integer' else (int,float)) and math.isfinite(value)
            if not valid:
                problems.append(f'{path}: not a finite {kind}')
            elif value<spec.get('minimum',-math.inf) or value>spec.get('maximum',math.inf):
                problems.append(f'{path}: out of range')

        elif kind=='string':
            if not isinstance(value,str):problems.append(f'{path}: not a string')
        elif kind=='boolean':
            if not isinstance(value,bool):problems.append(f'{path}: not a boolean')
        elif kind=='null':
            if value is not None:
                problems.append(f'{path}: not null')
        elif kind is not None:
            problems.append(f'{path}: unsupported schema type')
        if 'enum' in spec and value not in spec['enum']:problems.append(f'{path}: enum mismatch')
    check(output,schema,'$')
    return problems
